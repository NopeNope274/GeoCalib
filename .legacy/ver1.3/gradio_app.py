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
    "Sony VENICE 6K": {
        "sensor_w": 36.2, "sensor_h": 24.1,
        "modes": [
            {"name": "6K 3:2", "px_w": 6048, "px_h": 4032},
            {"name": "6K 17:9", "px_w": 6054, "px_h": 3192},
            {"name": "4K 17:9", "px_w": 4096, "px_h": 2160}
        ]
    },
    "ARRI Alexa 35": {
        "sensor_w": 27.99, "sensor_h": 19.22,
        "modes": [
            {"name": "4.6K 3:2 Open Gate", "px_w": 4608, "px_h": 3164},
            {"name": "4.6K 16:9", "px_w": 4608, "px_h": 2592},
            {"name": "4K 16:9", "px_w": 4096, "px_h": 2304}
        ]
    },
    "Full Frame (35mm)": {
        "sensor_w": 36.0, "sensor_h": 24.0,
        "modes": [{"name": "기본 3:2", "px_w": 6000, "px_h": 4000}]
    },
    "DJI Inspire 3 (Zenmuse X9-8K)": {
        "sensor_w": 40.51, "sensor_h": 21.36,
        "modes": [{"name": "8K 17:9", "px_w": 8192, "px_h": 4320}, {"name": "4K 16:9", "px_w": 3840, "px_h": 2160}]
    }
}

def load_presets():
    if not os.path.exists(PRESETS_FILE):
        save_presets(DEFAULT_PRESETS)
        return DEFAULT_PRESETS
    try:
        with open(PRESETS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 구버전 검사 (sensor_width 키로 확인) -> 구조 변경 시 강제 덮어쓰기
            if data and list(data.values())[0].get("sensor_width"):
                logger.warning("구버전 presets.json 발견. 새로운 계층형 구조로 초기화 덮어쓰기를 진행합니다.")
                save_presets(DEFAULT_PRESETS)
                return DEFAULT_PRESETS
            return data
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
custom_css = """
.scrollable-file-list {
    max-height: 250px;
    overflow-y: auto !important;
}
.copy-box input {
    font-weight: bold;
    color: #ff5722 !important;
}
"""

description = """
<p align="center">
  <h1 align="center"><ins>GeoCalib</ins> 📸<br>시퀀스로 카메라 캘리브레이션</h1>
</p>

## 📌 시작하기
이 도구는 **ECCV 2024 GeoCalib** 딥러닝 모델을 기반으로 드라마나 영화 촬영 소스에서 정확한 카메라 **화각(vFoV, hFoV)**과 **초점거리**를 추출합니다. 추출된 데이터는 **Maya, PFTrack, SynthEyes** 등으로 직접 붙여넣어 즉시 활용할 수 있습니다. 윈도우 환경 구조, 한글 경로, 대량의 이미지 시퀀스를 완벽 지원합니다.
"""

example_images = [
    [["assets/pinhole-church.jpg"]],
    [["assets/pinhole-garden.jpg"]],
    [["assets/fisheye-skyline.jpg"]],
    [["assets/fisheye-dog-pool.jpg"]],
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
    txt += "--- (아래 상세 수치는 첫 번째 이미지 기준 참고용) ---\n"
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

def process_results(image_paths, camera_model_korean, plot_up, plot_up_confidence, plot_latitude, plot_latitude_confidence, plot_undistort, progress=gr.Progress(track_tqdm=False)):
    if not image_paths:
        raise gr.Error("먼저 분석할 이미지를 하나 이상 업로드해주세요.")

    internal_camera_model = camera_model_choices[camera_model_korean]

    all_vfovs = []
    all_hfovs = []
    all_inference_results = []
    
    orig_w, orig_h = 0, 0
    total_imgs = len(image_paths)

    for i, single_image_path in enumerate(image_paths):
        progress((i) / total_imgs, desc=f"분석 중... [{i+1}/{total_imgs}]")
        
        img_tensor = model.load_image(single_image_path)
        current_inference_result = inference(img_tensor, internal_camera_model)

        if i == 0:
            orig_h, orig_w = img_tensor.shape[-2:]

        all_inference_results.append({
            "camera": current_inference_result["camera"],
            "gravity": current_inference_result["gravity"],
            "up_confidence": current_inference_result["up_confidence"],
            "latitude_confidence": current_inference_result["latitude_confidence"],
            "image_path": single_image_path
        })

        vfov_rad = current_inference_result["camera"].vfov.item()
        all_vfovs.append(vfov_rad)

        curr_h, curr_w = img_tensor.shape[-2:]
        hfov_rad = 2 * math.atan(math.tan(vfov_rad / 2) * (curr_w / curr_h))
        all_hfovs.append(hfov_rad)

    avg_vfov_rad = sum(all_vfovs) / len(all_vfovs)
    avg_hfov_rad = sum(all_hfovs) / len(all_hfovs)
    
    avg_vfov_deg = math.degrees(avg_vfov_rad)
    avg_hfov_deg = math.degrees(avg_hfov_rad)
    
    progress(1.0, desc="분석 완료 및 시각화 준비 중...")

    plot_img = render_frame(1, all_inference_results, plot_up, plot_up_confidence, plot_latitude, plot_latitude_confidence, plot_undistort)
    
    txt = format_output(all_inference_results[0], avg_vfov_deg, avg_hfov_deg, total_imgs)
    focal_px = all_inference_results[0]["camera"].f[0, 1].item()
    
    slider_update = gr.update(maximum=total_imgs, value=1, interactive=True, visible=total_imgs > 1)
    
    # Textboxes with copy buttons
    vfov_str = f"{avg_vfov_deg:.4f}"
    hfov_str = f"{avg_hfov_deg:.4f}"
    focal_str = f"{focal_px:.2f}"
    
    return txt, plot_img, all_inference_results, avg_vfov_deg, avg_hfov_deg, orig_w, orig_h, slider_update, vfov_str, hfov_str, focal_str

def render_frame(frame_idx, all_results, plot_up, plot_up_confidence, plot_latitude, plot_latitude_confidence, plot_undistort):
    """지정된 슬라이더 프레임 인덱스의 이미지를 실시간 로드하여 플로팅합니다."""
    if not all_results or frame_idx > len(all_results):
        return np.ones((128, 256, 3))
    
    res = all_results[int(frame_idx) - 1]
    img_tensor = model.load_image(res["image_path"])
    
    temp_res = {
        "camera": res["camera"],
        "gravity": res["gravity"],
        "up_confidence": res["up_confidence"],
        "latitude_confidence": res["latitude_confidence"],
        "image": img_tensor.cpu()
    }
    
    return update_plot(temp_res, plot_up, plot_up_confidence, plot_latitude, plot_latitude_confidence, plot_undistort)

def update_plot(inference_result, plot_up, plot_up_confidence, plot_latitude, plot_latitude_confidence, plot_undistort):
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
def save_markdown_report(result_text, all_results, avg_vfov_deg, avg_hfov_deg, w, h):
    if result_text is None or result_text.strip() == "" or not all_results:
        return "❌ 저장 실패: 분석된 결과가 없습니다. 먼저 캘리브레이션을 실행해주세요."

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    dir_path = filedialog.askdirectory(parent=root, title="마크다운(MD) 리포트 저장 폴더 선택", initialdir=os.path.expanduser("~"))
    root.destroy()

    if not dir_path:
        return "⚠️ 사용자 폴더 선택 취소됨."
        
    first_inference_result = all_results[0]
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

# ----------------- 5. Camera Math Calculator (Cascades) ----------------- 
def get_mode_choices_and_initial(preset_name):
    presets = load_presets()
    if not preset_name or preset_name not in presets:
        return gr.update(choices=["해당사항 없음"], value="해당사항 없음", interactive=False)
    modes = presets[preset_name].get("modes", [])
    if not modes:
        return gr.update(choices=["단일 모델 (기본값)"], value="단일 모델 (기본값)", interactive=False)
        
    names = [m["name"] for m in modes]
    return gr.update(choices=names, value=names[0], interactive=True)

def fill_px_from_mode(preset_name, mode_name):
    presets = load_presets()
    if not preset_name or preset_name not in presets:
        return gr.update(), gr.update()
    modes = presets[preset_name].get("modes", [])
    for m in modes:
        if m["name"] == mode_name:
            return gr.update(value=str(m["px_w"])), gr.update(value=str(m["px_h"]))
    return gr.update(), gr.update()

def update_calculator_auto(preset_name, mode_name, orig_w, orig_h, cur_w, cur_h, state_vfov_deg):
    if state_vfov_deg is None or state_vfov_deg == 0:
        return "오류: 먼저 분석 탭에서 분석을 완료해 vFoV 값을 도출하세요.", None
        
    presets = load_presets()
    if preset_name not in presets:
        return f"오류: 프리셋 '{preset_name}'을 찾을 수 없습니다.", None
        
    sensor_w = presets[preset_name]["sensor_w"]
    sensor_h = presets[preset_name]["sensor_h"]
    
    try:
        orig_h, cur_h = float(orig_h), float(cur_h)
        zoom_y = cur_h / orig_h 
        
        effective_sensor_h = sensor_h * zoom_y
        vfov_rad = math.radians(state_vfov_deg)
        focal_length_mm = effective_sensor_h / (2 * math.tan(vfov_rad / 2))
        
        mode_str = f"[{mode_name}] " if mode_name and mode_name != "단일 모델 (기본값)" else ""
        
        txt = f"[{preset_name}] {mode_str}기준 원본 센서 해상도 매칭.\n"
        txt += f"원본대비 시퀀스 크롭/리전 비율 체크 (Y축 높이 비례 {zoom_y*100:.1f}% 활용됨).\n"
        txt += f"유효 공간 센서 세로 높이: {effective_sensor_h:.3f}mm\n"
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
def preset_manager_add(name, w, h, df_data):
    presets = load_presets()
    try:
        sw, sh = float(w), float(h)
    except:
         return gr.update(), "오류: 센서 크기는 숫자여야 합니다."
    if not name:
         return gr.update(), "오류: 프리셋 이름을 입력하세요."
         
    modes = []
    # parse the array returned by gr.Dataframe(type='array')
    for row in df_data:
        m_name = str(row[0]).strip()
        if m_name and str(row[0]).lower() != "nan":
            try:
                pw, ph = float(row[1]), float(row[2])
                modes.append({"name": m_name, "px_w": pw, "px_h": ph})
            except:
                pass
                
    presets[name] = {"sensor_w": sw, "sensor_h": sh, "modes": modes}
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
    state_all_results = gr.State([])

    with gr.Tabs():
        # ======= TAB 1 =======
        with gr.Tab("📷 Tab 1: 분석 및 출력 (Analysis)"):
            with gr.Row():
                # 좌측 컴포넌트들
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 이미지 소스 로드 및 모델 선택")
                    file_input = gr.File(label="이미지 시퀀스 업로드 (다중 선택/한글 폴더 완벽 대응)", file_count="multiple", file_types=["image"], elem_classes="scrollable-file-list")
                    choice_input = gr.Dropdown(
                        choices=list(camera_model_choices.keys()), 
                        label="카메라 렌즈 모델 알고리즘 선택", 
                        value=list(camera_model_choices.keys())[1]
                    )
                    
                    submit_btn = gr.Button("🚀 캘리브레이션 실행 (GPU 연산)", variant="primary")
                    
                    gr.Markdown("---")
                    gr.Markdown("### 2. 구동 결과 저장 시스템")
                    save_btn = gr.Button("💾 결과 값 저장")
                    save_out = gr.Textbox(label="결과 출력", interactive=False)
                    
                    gr.Markdown("---")
                    gr.Markdown("### ※ 사용법 예제 레퍼런스")
                    gr.Examples(examples=example_images, inputs=[file_input, choice_input])
                
                # 우측 컴포넌트들
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 분석 결과")
                    with gr.Row():
                        out_vfov = gr.Textbox(label="vFoV (°)", elem_classes="copy-box", interactive=False)
                        out_hfov = gr.Textbox(label="hFoV (°)", elem_classes="copy-box", interactive=False)
                        out_focalpx = gr.Textbox(label="Focal Length (px)", elem_classes="copy-box", interactive=False)
                        
                    # Tracking Guide Text
                    gr.Markdown("""
                    ### 🎯 트래킹 소프트웨어 연동 가이드
                    * **PFTrack 2020+**: `Edit Camera` 설정창의 `Focal Length (Pixels)` 옵션에 복사한 **[Focal Length (px)]** 를 바로 입력하세요. _(Physical mm계산불필요)_
                    * **SynthEyes**: `Image Preparation`의 렌즈 솔버 설정 시 **[hFoV (수평화각)]** 을 주로 사용하며, 또는 Image 기반 포맷으로 **[Focal Length (px)]** 를 그대로 활용하세요.
                    * **Maya**: 제작된 Camera의 `Angle of View (Vertical)` 어트리뷰트에 **[vFoV (수직화각)]**를 입력하면 프레임 화각이 즉시 매칭됩니다.
                    """)
                    
                    text_output = gr.Textbox(label="결과 값 정리 (시퀀스 회전 및 왜곡 파라미터)", lines=4)
                    
                    image_output = gr.Image(label="렌즈 시각화 결과 (인터랙티브 뷰어)")
                    frame_slider = gr.Slider(minimum=1, maximum=1, step=1, value=1, label="프레임 슬라이더 (좌우로 움직여 정합성 검증)", interactive=True, visible=False)
                    
                    gr.Markdown("#### Visual 가이드 옵션")
                    with gr.Row():
                        plot_up = gr.Checkbox(label="가로형 필드(Up-vector) 표시", value=True)
                        plot_latitude = gr.Checkbox(label="위도(Latitude) 필드 표시", value=True)
                    with gr.Row():
                        plot_up_confidence = gr.Checkbox(label="Up-vector 신뢰 구간", value=False)
                        plot_latitude_confidence = gr.Checkbox(label="위도 신뢰 구간", value=False)
                    plot_undistort = gr.Checkbox(label="Undistort", value=False)

        # ======= TAB 2 =======
        with gr.Tab("🧮 Tab 2: 렌즈 초점거리 계산기 (Camera Calculator)"):
            gr.Markdown("분석 탭에서 도출된 **평균 수직 화각(vFoV)** 을 바탕으로 실제 물리 렌즈 초점거리(mm)를 산출해 줍니다.\n**CG 툴**의 Camera Solver에 실제 Lens mm 정보를 필수로 입력해야 할 때 유용합니다. (독립적으로 작동합니다)")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🤖 자동 모드 (프리셋 & 촬영 크롭비율 기반)")
                    preset_choices = list(load_presets().keys())
                    calc_preset_dropdown = gr.Dropdown(choices=preset_choices, label="카메라 바디(센서) 기종 선택", value=preset_choices[0] if preset_choices else None)
                    calc_mode_dropdown = gr.Dropdown(choices=[], label="해상도 촬영 모드 (크롭/오픈게이트)", interactive=True)
                    
                    with gr.Row():
                        calc_orig_w = gr.Textbox(label="가로 원본 px", value="")
                        calc_orig_h = gr.Textbox(label="세로 원본 px", value="")
                    with gr.Row():
                        calc_cur_w = gr.Textbox(label="가로 현재 시퀀스 px", value="1920")
                        calc_cur_h = gr.Textbox(label="세로 현재 시퀀스 px", value="1080")
                        
                    calc_auto_btn = gr.Button("자동 계산 실행", variant="primary")
                    calc_auto_result = gr.Textbox(label="자동 환산 결과 로그", lines=4)
                    
                with gr.Column():
                    gr.Markdown("### ⚙️ 수동 모드 (직접 센서 사이즈 기입)")
                    calc_manual_w = gr.Textbox(label="센서 가로 길이 (mm)", value="36.0")
                    calc_manual_h = gr.Textbox(label="센서 세로 길이 (mm)", value="24.0")
                    calc_man_btn = gr.Button("수동 방식 실행", variant="secondary")
                    calc_man_result = gr.Textbox(label="수동 환산 결과 로그", lines=2)
            
            gr.Markdown("---")
            gr.Markdown("### 📝 카메라 센서 프리셋 매니저 (계층형 추가/삭제)")
            with gr.Row():
                with gr.Column(scale=1):
                    mgr_name = gr.Textbox(label="새 프리셋 이름 지정 (예: RED V-Raptor 8K VV)")
                    mgr_w = gr.Textbox(label="센서 가로 치수 (mm)")
                    mgr_h = gr.Textbox(label="센서 세로 치수 (mm)")
                with gr.Column(scale=2):
                    gr.Markdown("#### 등록할 해상도 모드 추가 (+ 버튼 클릭하여 열 추가)")
                    mgr_modes_df = gr.Dataframe(headers=["모드 이름", "가로 해상도(px)", "세로 해상도(px)"], datatype=["str", "number", "number"], col_count=(3, "fixed"), row_count=1, interactive=True, type="array")

            with gr.Row():
                mgr_add_btn = gr.Button("✅ 리스트 최신화 (추가/수정)")
                mgr_del_btn = gr.Button("🗑️ 선택된 프리셋 삭제 로직 진행", variant="stop")
            mgr_msg = gr.Textbox(label="관리 결과 상태창")

        # ======= TAB 3 =======
        with gr.Tab("📖 Tab 3: 도움말 및 환경 설치 안내"):
            gr.Markdown("""
## 🚀 포터블 파이썬 환경 세팅 요령
MBC 팀원 등 Anaconda와 Git 설치가 번거로운 타 직군이나 외주 인력이 **이 폴더를 그대로 타 윈도우 PC로 가져가서 클릭 한 번에 구동할 수 있는 법**입니다.

1. **포터블 파이썬 다운로드**: 파이썬 공식 사이트에서 `Windows embeddable package (64-bit)` 버전을 다운로드 받아 이 저장소 폴더(GeoCalib 폴더 안)의 `python` 폴더 압축을 풉니다.
2. **Setup 환경 구축**: `Setup_Environment.bat` 배치 파일을 아래 내용을 넣어 만드세요.
```bat
@echo off
echo 파이썬 패키지를 설치합니다...
.\\python\\python.exe -m pip install -r requirements.txt
echo 설치가 완료되었습니다.
pause
```
3. 라이브러리 설치가 끝나면, 앞으로 어떤 PC에서든 `Start_GeoCalib.bat` 만 더블클릭하면 즉각 실행됩니다.

## ⚙️ 간편 실행법 (Start_GeoCalib.bat)
- `Start_GeoCalib.bat` 내부에는 로컬 가상환경(venv 혹은 miniconda env)을 활성화하거나 `python gradio_app.py`를 실행하도록 세팅되어 있습니다.
- 백그라운드 콘솔창에 오류/경고(`[geocalib.lm_optimizer WARNING]`)가 뜨며 수십초 멈춰있더라도, 이는 렌즈 기하학 정합성을 연산하는 내부의 정상적인 루틴이므로 기다려주시면 됩니다. 완료 즉시 크롬 등 기본 브라우저 창에서 UI가 열립니다.
            """)

    # ----------------- 8. Event Binding ----------------- 
    plot_inputs = [
        frame_slider, state_all_results, plot_up, plot_up_confidence, plot_latitude, plot_latitude_confidence, plot_undistort
    ]
    
    # 분석 실행
    submit_btn.click(
        fn=process_results,
        inputs=[file_input, choice_input, plot_up, plot_up_confidence, plot_latitude, plot_latitude_confidence, plot_undistort],
        outputs=[text_output, image_output, state_all_results, state_vfov, state_hfov, state_orig_w, state_orig_h, frame_slider, out_vfov, out_hfov, out_focalpx],
    )

    # 슬라이더 이동 시 원본 로드 및 실시간 플로팅
    frame_slider.change(
        fn=render_frame,
        inputs=plot_inputs,
        outputs=image_output
    )

    # 마크다운 저장
    save_btn.click(
        fn=save_markdown_report,
        inputs=[text_output, state_all_results, state_vfov, state_hfov, state_orig_w, state_orig_h],
        outputs=save_out
    )

    # 플롯 옵션 변경 시 업데이트
    plot_up.change(fn=render_frame, inputs=plot_inputs, outputs=image_output)
    plot_up_confidence.change(fn=render_frame, inputs=plot_inputs, outputs=image_output)
    plot_latitude.change(fn=render_frame, inputs=plot_inputs, outputs=image_output)
    plot_latitude_confidence.change(fn=render_frame, inputs=plot_inputs, outputs=image_output)
    plot_undistort.change(fn=render_frame, inputs=plot_inputs, outputs=image_output)

    # Tab 2 Events (계산기 계층형 드롭다운)
    demo.load(fn=get_mode_choices_and_initial, inputs=[calc_preset_dropdown], outputs=[calc_mode_dropdown])
    calc_preset_dropdown.change(fn=get_mode_choices_and_initial, inputs=[calc_preset_dropdown], outputs=[calc_mode_dropdown])
    calc_mode_dropdown.change(fn=fill_px_from_mode, inputs=[calc_preset_dropdown, calc_mode_dropdown], outputs=[calc_orig_w, calc_orig_h])

    calc_auto_btn.click(
        fn=update_calculator_auto,
        inputs=[calc_preset_dropdown, calc_mode_dropdown, calc_orig_w, calc_orig_h, calc_cur_w, calc_cur_h, state_vfov],
        outputs=[calc_auto_result, gr.State()]
    )

    calc_man_btn.click(
        fn=update_calculator_manual,
        inputs=[calc_manual_w, calc_manual_h, state_vfov],
        outputs=[calc_man_result, gr.State()]
    )

    # Preset Manager Events
    mgr_add_btn.click(
        fn=preset_manager_add,
        inputs=[mgr_name, mgr_w, mgr_h, mgr_modes_df],
        outputs=[calc_preset_dropdown, mgr_msg]
    )

    mgr_del_btn.click(
        fn=preset_manager_delete,
        inputs=[calc_preset_dropdown],
        outputs=[calc_preset_dropdown, mgr_msg]
    )

# Launch
if __name__ == "__main__":
    demo.launch(css=custom_css)
