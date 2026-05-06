"""Gradio app for GeoCalib inference. Redesigned for MBC Design Team."""

import os
import sys
import json
import math
import tkinter as tk
from tkinter import filedialog
from time import time
import datetime
from copy import deepcopy

import gradio as gr
import numpy as np
import torch
import cv2

from geocalib import logger, viz2d
from geocalib.camera import camera_models
from geocalib.extractor import GeoCalib
from geocalib.perspective_fields import get_perspective_field
from geocalib.utils import rad2deg

# flake8: noqa
# mypy: ignore-errors

# ----------------- 1. Preset Manager ----------------- 
PRESETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.json")

DEFAULT_PRESETS = {
    "Sony VENICE 6K (3:2)": {"sensor_width": 36.2, "sensor_height": 24.1},
    "ARRI Alexa 35": {"sensor_width": 27.99, "sensor_height": 19.22},
    "Full Frame (35mm)": {"sensor_width": 36.0, "sensor_height": 24.0},
    "APS-C (1.5x)": {"sensor_width": 23.6, "sensor_height": 15.6},
    "Micro Four Thirds (MFT)": {"sensor_width": 17.3, "sensor_height": 13.0}
}

def load_presets():
    if not os.path.exists(PRESETS_FILE):
        with open(PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_PRESETS, f, indent=4, ensure_ascii=False)
        return DEFAULT_PRESETS
    try:
        with open(PRESETS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"프리셋 로딩 실패: {e}")
        return DEFAULT_PRESETS

def save_presets(presets_dict):
    try:
        with open(PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(presets_dict, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"프리셋 저장 실패: {e}")
        return False

# ----------------- 2. System Settings ----------------- 
description = """
<p align="center">
  <h1 align="center"><ins>GeoCalib</ins> 📸<br>단일 이미지 카메라 캘리브레이션 툴 (MBC 디자인팀 최적화)</h1>
</p>

## 📌 시작하기
이 도구는 **ECCV 2024 GeoCalib** 딥러닝 모델을 기반으로 드라마나 영화 촬영 소스에서 정확한 카메라 **화각(vFoV, hFoV)**과 **초점거리**를 추출합니다. 추출된 데이터는 **Maya, PFTrack, SynthEyes** 등으로 직접 붙여넣어 즉시 활용할 수 있습니다. 윈도우 환경 구조, 한글 경로, 대량의 이미지 시퀀스를 완벽 지원합니다.
"""

example_images = [
    ["assets/pinhole-church.jpg"],
    ["assets/pinhole-garden.jpg"],
    ["assets/fisheye-skyline.jpg"],
    ["assets/fisheye-dog-pool.jpg"],
]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoCalib().to(device)

camera_model_choices = {
    "핀홀 (왜곡 없음 / CG 렌더링 소스용)": "pinhole",
    "단순 방사형 (일반적인 렌즈 왜곡 / 대부분의 드라마·영화 추천)": "simple_radial",
    "방사형 (정교한 왜곡 / 광각 렌즈용)": "radial",
    "단순 분할형 (심한 왜곡 / 어안 렌즈용)": "simple_divisional",
}

# ----------------- 3. Core Engine Logic ----------------- 
def format_output(results, avg_vfov, avg_hfov, seq_count):
    camera, gravity = results["camera"], results["gravity"]
    roll, pitch = rad2deg(gravity.rp).unbind(-1)

    txt = "📌 [분석 완료]\n"
    txt += f"- 분석된 시퀀스 이미지 수: {seq_count}장\n"
    txt += f"- 평균 수직 화각 (vFoV): {avg_vfov:.4f}°\n"
    txt += f"- 평균 수평 화각 (hFoV): {avg_hfov:.4f}°\n"
    txt += "--- (아래 상세 수치는 첫 번째 이미지 기준 트래킹 앱 수동 입력 참조용) ---\n"
    txt += f"- 초점 거리 (Focal Length px): {camera.f[0, 1].item():.2f} px\n"
    txt += f"- 롤 (Roll): {roll.item():.2f}°\n"
    txt += f"- 피치 (Pitch): {pitch.item():.2f}°\n"
    if hasattr(camera, "k1"):
        txt += f"- 렌즈 왜곡 계수 (K1): {camera.k1[0].item():.4f}\n"
    return txt

def inference(img_tensor, camera_model):
    out = model.calibrate(img_tensor.to(device), camera_model=camera_model)
    save_keys = ["camera", "gravity"] + [
        f"{k}_uncertainty" for k in ["roll", "pitch", "vfov", "focal"]
    ]
    res = {k: v.cpu() for k, v in out.items() if k in save_keys}
    res["up_confidence"] = out["up_confidence"].cpu().numpy()
    res["latitude_confidence"] = out["latitude_confidence"].cpu().numpy()
    return res

def process_results(image_paths, camera_model_korean, plot_up, plot_up_confidence, plot_latitude, plot_latitude_confidence, plot_undistort):
    if not image_paths:
        raise gr.Error("먼저 분석할 이미지를 하나 이상 업로드해주세요. 한글이 포함된 경로도 정상 지원됩니다.")

    internal_camera_model = camera_model_choices[camera_model_korean]

    all_vfovs = []
    all_hfovs = []
    first_inference_result = None
    orig_w, orig_h = 0, 0

    for i, single_image_path in enumerate(image_paths):
        # geocalib내의 model.load_image는 utils.np.fromfile를 통해 이미 안전하게 한글경로 바이트 처리를 합니다.
        img_tensor = model.load_image(single_image_path)
        
        start = time()
        current_inference_result = inference(img_tensor, internal_camera_model)

        if i == 0:
            first_inference_result = deepcopy(current_inference_result)
            first_inference_result["image"] = img_tensor.cpu()
            orig_h, orig_w = img_tensor.shape[-2:]

        # vFoV 수치 (라디안)
        vfov_rad = current_inference_result["camera"].vfov.item()
        all_vfovs.append(vfov_rad)

        # hFoV 수치 (라디안) 공식 적용
        curr_h, curr_w = img_tensor.shape[-2:]
        hfov_rad = 2 * math.atan(math.tan(vfov_rad / 2) * (curr_w / curr_h))
        all_hfovs.append(hfov_rad)

    # 전체 시퀀스 평균 (라디안 -> 디그리)
    avg_vfov_rad = sum(all_vfovs) / len(all_vfovs)
    avg_hfov_rad = sum(all_hfovs) / len(all_hfovs)
    
    avg_vfov_deg = math.degrees(avg_vfov_rad)
    avg_hfov_deg = math.degrees(avg_hfov_rad)

    plot_img = update_plot(
        first_inference_result, plot_up, plot_up_confidence,
        plot_latitude, plot_latitude_confidence, plot_undistort
    )

    txt = format_output(first_inference_result, avg_vfov_deg, avg_hfov_deg, len(image_paths))
    return txt, plot_img, first_inference_result, avg_vfov_deg, avg_hfov_deg, orig_w, orig_h

def update_plot(inference_result, plot_up, plot_up_confidence, plot_latitude, plot_latitude_confidence, plot_undistort):
    if inference_result is None:
        gr.Error("먼저 이미지를 캘리브레이션해주세요.")
        return np.ones((128, 256, 3))

    camera, gravity = inference_result["camera"], inference_result["gravity"]
    img = inference_result["image"].permute(1, 2, 0).numpy()

    if plot_undistort:
        if not hasattr(camera, "k1"):
            return img
        return camera.undistort_image(inference_result["image"][None])[0].permute(1, 2, 0).numpy()

    up, lat = get_perspective_field(camera, gravity)
    fig = viz2d.plot_images([img], pad=0)
    ax = fig.get_axes()

    if plot_up:
        viz2d.plot_vector_fields([up[0]], axes=[ax[0]])
    if plot_latitude:
        viz2d.plot_latitudes([lat[0, 0]], axes=[ax[0]])
    if plot_up_confidence:
        viz2d.plot_confidences([torch.tensor(inference_result["up_confidence"][0])], axes=[ax[0]])
    if plot_latitude_confidence:
        viz2d.plot_confidences([torch.tensor(inference_result["latitude_confidence"][0])], axes=[ax[0]])

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    return img

# ----------------- 4. Output / File Management ----------------- 
def save_markdown_report(result_text, first_inference_result, avg_vfov_deg, avg_hfov_deg, w, h):
    """지정된 폴더에 마크다운 파일로 결과 저장."""
    if result_text is None or result_text.strip() == "" or first_inference_result is None:
        return "❌ 저장 실패: 분석된 결과가 없습니다. 먼저 캘리브레이션을 실행해주세요."

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) # 창이 숨지 않도록 방지
    
    dir_path = filedialog.askdirectory(parent=root, title="마크다운(MD) 리포트 저장 폴더 선택", initialdir=os.path.expanduser("~"))
    root.destroy()

    if not dir_path:
        return "⚠️ 사용자 폴더 선택 취소됨."
        
    camera = first_inference_result["camera"]
    focal_length_px = camera.f[0, 1].item()
    roll, pitch = rad2deg(first_inference_result["gravity"].rp).unbind(-1)
    k1_str = f"{camera.k1[0].item():.4f}" if hasattr(camera, "k1") else "없음 (Pinhole)"

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"GeoCalib_Report_{now_str}.md"
    save_path = os.path.join(dir_path, file_name)

    md_content = f"""# 🎥 GeoCalib Camera Analysis Report
**생성 일시:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**기준 해상도:** {w} x {h} px

## 📊 핵심 카메라/렌즈 데이터 (PFTrack & Maya 추천 수치)
| 파라미터 (Parameter) | 추출값 (Value) | 비고 (Remarks) |
| :--- | :--- | :--- |
| **수직 화각 (vFoV)** | `{avg_vfov_deg:.4f}°` | Maya Camera의 **Angle of View (Vertical)** 입력에 권장. |
| **수평 화각 (hFoV)** | `{avg_hfov_deg:.4f}°` | PFTrack / 일반적 H-FOV 입력 기준. |
| **초점 거리 (Focal Length px)** | `{focal_length_px:.2f} px` | PFTrack 2020의 **Pixel Focal Length** 수동 입력 시 활용. |

## 🛠 왜곡 및 회전 (Distortion & Rotation) - 첫번째 이미지 기준
| 요소 | 추정값 | 
| :--- | :--- |
| **롤 (Roll)** | `{roll.item():.2f}°` |
| **피치 (Pitch)** | `{pitch.item():.2f}°` |
| **왜곡 계수 (Radial K1)** | `{k1_str}` |

---
*Generated by **MBC Design Team 파이프라인 TD***
"""
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        return f"✅ 성공: {save_path} 저장완료!"
    except Exception as e:
        return f"❌ 저장 중 오류 발생: {str(e)}"

# ----------------- 5. Camera Math Calculator ----------------- 
def update_calculator_auto(preset_name, orig_w, orig_h, cur_w, cur_h, state_vfov_deg):
    if state_vfov_deg is None or state_vfov_deg == 0:
        return "오류: 먼저 분석 탭에서 분석을 완료해 vFoV 값을 도출하세요.", None
        
    presets = load_presets()
    if preset_name not in presets:
        return f"오류: 프리셋 '{preset_name}'을 찾을 수 없습니다.", None
        
    sensor_w = presets[preset_name]["sensor_width"]
    sensor_h = presets[preset_name]["sensor_height"]
    
    try:
        orig_h, cur_h = float(orig_h), float(cur_h)
        zoom_y = cur_h / orig_h 
        
        effective_sensor_h = sensor_h * zoom_y
        vfov_rad = math.radians(state_vfov_deg)
        focal_length_mm = effective_sensor_h / (2 * math.tan(vfov_rad / 2))
        
        txt = f"[{preset_name}] 기준 원본 센서 해상도 매칭.\n"
        txt += f"원본대비 시퀀스 크롭/리전 비율 체크 (Y축 기준 높이 {zoom_y*100:.1f}% 활용됨).\n"
        txt += f"유효 공간의 센서 세로 높이: {effective_sensor_h:.3f}mm\n"
        txt += f"👉 **실제 피지컬 초점 거리 (Focal Length): {focal_length_mm:.2f} mm**"
        
        return txt, focal_length_mm
    except ValueError:
        return "오류: 해상도에는 올바른 숫자를 입력해주세요.", None

def update_calculator_manual(manual_sensor_w, manual_sensor_h, state_vfov_deg):
    if state_vfov_deg is None or state_vfov_deg == 0:
        return "오류: 먼저 분석 탭에서 분석을 완료해 vFoV 값을 도출하세요.", None
    try:
        sh = float(manual_sensor_h)
        vfov_rad = math.radians(state_vfov_deg)
        focal_length_mm = sh / (2 * math.tan(vfov_rad / 2))
        
        txt = f"입력된 센서 세로 높이: {sh}mm (분석된 평균 vFoV 기준 정산)\n"
        txt += f"👉 **비례 초점 거리 (Focal Length): {focal_length_mm:.2f} mm**"
        
        return txt, focal_length_mm
    except ValueError:
        return "오류: 센서 크기에 숫자를 올바르게 입력해주세요.", None


# ----------------- 6. Preset Manager Frontend ----------------- 
def preset_manager_add(name, w, h):
    presets = load_presets()
    try:
        sw, sh = float(w), float(h)
    except:
         return gr.update(), "오류: 센서 크기는 숫자여야 합니다."
    if not name:
         return gr.update(), "오류: 프리셋 이름을 입력하세요."
         
    presets[name] = {"sensor_width": sw, "sensor_height": sh}
    save_presets(presets)
    keys = list(presets.keys())
    return gr.update(choices=keys, value=name), f"'{name}' 추가 및 저장 시스템 동기화 완료."
    
def preset_manager_delete(name):
    presets = load_presets()
    if name in presets:
         del presets[name]
         save_presets(presets)
         keys = list(presets.keys())
         val = keys[0] if keys else None
         return gr.update(choices=keys, value=val), f"'{name}' 삭제 완료."
    return gr.update(), "오류: 해당 프리셋이 존재하지 않습니다."

# ----------------- 7. UI Definition ----------------- 
with gr.Blocks(title="GeoCalib for MBC Design Team") as demo:
    gr.Markdown(description)

    # State stores
    state_vfov = gr.State(0.0)
    state_hfov = gr.State(0.0)
    state_orig_w = gr.State(0.0)
    state_orig_h = gr.State(0.0)
    state_first_result = gr.State(None)

    with gr.Tabs():
        # ======= TAB 1 =======
        with gr.Tab("📷 Tab 1: 카메라 분석 및 산출 (Analysis)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 이미지 소스 로드 및 모델 선택")
                    file_input = gr.File(label="이미지 시퀀스 업로드 (다중 선택/한글 폴더 완벽 대응)", file_count="multiple", file_types=["image"])
                    choice_input = gr.Dropdown(
                        choices=list(camera_model_choices.keys()), 
                        label="카메라 렌즈 모델 알고리즘 선택", 
                        value=list(camera_model_choices.keys())[1]
                    )
                    
                    submit_btn = gr.Button("🚀 캘리브레이션 실행 (GPU 연산)", variant="primary")
                    
                    gr.Markdown("---")
                    gr.Markdown("### 2. 결과 출력 및 Markdown 엑스포트")
                    save_btn = gr.Button("💾 윈도우 탐색기 열기 (마크다운 표 형식 저장)")
                    save_out = gr.Textbox(label="결과 저장 상태 표시", interactive=False)
                    
                    gr.Markdown("---")
                    gr.Markdown("### ※ 사용법 예매 레퍼런스")
                    gr.Examples(examples=example_images, inputs=[file_input, choice_input])
                    
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 분석 도출 요약 (PFTrack / Maya 입력 데이터)")
                    text_output = gr.Textbox(label="시퀀스 추정 파라미터 텍스트 결과창", lines=8)
                    image_output = gr.Image(label="렌즈 시각화 결과 (시퀀스 첫 번째 이미지 플롯 기준)")
                    
                    gr.Markdown("#### 시각적 플롯 렌더링 옵션")
                    with gr.Row():
                        plot_up = gr.Checkbox(label="가로형 필드(Up-vector) 표시", value=True)
                        plot_latitude = gr.Checkbox(label="위도(Latitude) 필드 표시", value=True)
                    with gr.Row():
                        plot_up_confidence = gr.Checkbox(label="Up-vector 신뢰 구간", value=False)
                        plot_latitude_confidence = gr.Checkbox(label="위도 신뢰 구간", value=False)
                    plot_undistort = gr.Checkbox(label="렌즈 왜곡 보정 펼치기 (다른 시각화를 덮어씀)", value=False)

        # ======= TAB 2 =======
        with gr.Tab("🧮 Tab 2: 렌즈 초점거리 계산기 (Camera Calculator)"):
            gr.Markdown("분석 탭에서 도출된 **평균 수직 화각(vFoV)** 을 바탕으로 실제 물리 렌즈 초점거리(mm)를 산출해 줍니다.\n**CG 툴**의 Camera Solver에 실제 렌즈 정보를 입력해야 할 때 유용합니다.")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🤖 자동 모드 (프리셋 & 촬영 크롭비율 기반)")
                    preset_choices = list(load_presets().keys())
                    calc_preset_dropdown = gr.Dropdown(choices=preset_choices, label="카메라 바디(센서) 기종 선택", value=preset_choices[0] if preset_choices else None)
                    
                    gr.Markdown("원본 촬영 해상도와 포스트 컴프 해상도가 다르다면 비례 크롭이 자동 계산됩니다.")
                    with gr.Row():
                        calc_orig_w = gr.Textbox(label="가로(Width) 원본 px", value="6000")
                        calc_orig_h = gr.Textbox(label="세로(Height) 원본 px", value="4000")
                    with gr.Row():
                        calc_cur_w = gr.Textbox(label="가로(Width) 현재 시퀀스 px", value="1920")
                        calc_cur_h = gr.Textbox(label="세로(Height) 현재 시퀀스 px", value="1080")
                        
                    calc_auto_btn = gr.Button("자동 계산 실행", variant="primary")
                    calc_auto_result = gr.Textbox(label="자동 환산 결과 로그", lines=4)
                    
                with gr.Column():
                    gr.Markdown("### ⚙️ 수동 모드 (직접 센서 사이즈 기입)")
                    calc_manual_w = gr.Textbox(label="센서 가로 길이 (mm)", value="36.0")
                    calc_manual_h = gr.Textbox(label="센서 세로 길이 (mm)", value="24.0")
                    calc_man_btn = gr.Button("수동 방식 실행", variant="secondary")
                    calc_man_result = gr.Textbox(label="수동 환산 결과 로그", lines=2)
            
            gr.Markdown("---")
            gr.Markdown("### 📝 카메라 센서 프리셋 매니저 (추가/삭제)")
            with gr.Row():
                mgr_name = gr.Textbox(label="새 프리셋 이름 지정 (예: RED V-Raptor 8K VV)")
                mgr_w = gr.Textbox(label="센서 가로 치수 (mm)")
                mgr_h = gr.Textbox(label="센서 세로 치수 (mm)")
            with gr.Row():
                mgr_add_btn = gr.Button("✅ 리스트 최신화 (추가/수정)")
                mgr_del_btn = gr.Button("🗑️ 선택된 프리셋 삭제 로직 진행", variant="stop")
            mgr_msg = gr.Textbox(label="관리 결과 상태창")

    # ----------------- 8. Event Binding ----------------- 
    # Tab 1 Events
    plot_inputs = [
        state_first_result, plot_up, plot_up_confidence,
        plot_latitude, plot_latitude_confidence, plot_undistort,
    ]
    
    submit_btn.click(
        fn=process_results,
        inputs=[file_input, choice_input, plot_up, plot_up_confidence, plot_latitude, plot_latitude_confidence, plot_undistort],
        outputs=[text_output, image_output, state_first_result, state_vfov, state_hfov, state_orig_w, state_orig_h],
    )

    save_btn.click(
        fn=save_markdown_report,
        inputs=[text_output, state_first_result, state_vfov, state_hfov, state_orig_w, state_orig_h],
        outputs=save_out
    )

    plot_up.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)
    plot_up_confidence.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)
    plot_latitude.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)
    plot_latitude_confidence.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)
    plot_undistort.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)

    # Tab 2 Events
    calc_auto_btn.click(
        fn=update_calculator_auto,
        inputs=[calc_preset_dropdown, calc_orig_w, calc_orig_h, calc_cur_w, calc_cur_h, state_vfov],
        outputs=[calc_auto_result, gr.State()]
    )

    calc_man_btn.click(
        fn=update_calculator_manual,
        inputs=[calc_manual_w, calc_manual_h, state_vfov],
        outputs=[calc_man_result, gr.State()]
    )

    mgr_add_btn.click(
        fn=preset_manager_add,
        inputs=[mgr_name, mgr_w, mgr_h],
        outputs=[calc_preset_dropdown, mgr_msg]
    )

    mgr_del_btn.click(
        fn=preset_manager_delete,
        inputs=[calc_preset_dropdown],
        outputs=[calc_preset_dropdown, mgr_msg]
    )

# Launch
if __name__ == "__main__":
    demo.launch()
