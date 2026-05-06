"""Gradio app for GeoCalib inference."""

from copy import deepcopy
from time import time

import os
import gradio as gr
import numpy as np
# import spaces
import torch

from geocalib import logger, viz2d
from geocalib.camera import camera_models
from geocalib.extractor import GeoCalib
from geocalib.perspective_fields import get_perspective_field
from geocalib.utils import rad2deg

# flake8: noqa
# mypy: ignore-errors

description = """
<p align="center">
  <h1 align="center"><ins>GeoCalib</ins> 📸<br>기하학적 최적화를 이용한 단일 이미지 카메라 캘리브레이션</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/alexander-veicht/">Alexander Veicht</a>
    ·
    <a href="https://psarlin.com/">Paul-Edouard&nbsp;Sarlin</a>
    ·
    <a href="https://www.linkedin.com/in/philipplindenberger/">Philipp Lindenberger</a>
    ·
    <a href="https://www.microsoft.com/en-us/research/people/mapoll/">Marc&nbsp;Pollefeys</a>
  </p>
  <h2 align="center">
    <p>ECCV 2024</p>
    <a href="https://arxiv.org/pdf/2409.06704" align="center">논문</a> |
    <a href="https://github.com/cvg/GeoCalib" align="center">코드</a> |
    <a href="https://colab.research.google.com/drive/1oMzgPGppAPAIQxe-s7SRd_q8r7dVfnqo#scrollTo=etdzQZQzoo-K" align="center">Colab</a>
  </h2>
</p>

## 시작하기
GeoCalib은 딥러닝과 기하학적 최적화를 결합하여 단일 이미지로부터 카메라 내부 파라미터(Intrinsics)와 중력 방향을 정확하게 추정합니다.

시작하려면 이미지를 업로드하거나 아래 예시 중 하나를 선택하세요.
다양한 카메라 모델을 선택하고 캘리브레이션 결과를 시각화할 수 있습니다.

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


def format_output(results, avg_vfov):
    camera, gravity = results["camera"], results["gravity"]
    # vfov = rad2deg(camera.vfov) # 이제 평균 vFoV를 사용합니다.
    roll, pitch = rad2deg(gravity.rp).unbind(-1)

    txt = "추정된 파라미터:\n"
    txt += f"평균 수직 화각(vFoV): {avg_vfov:.2f}°\n"
    txt += f"--- (아래는 첫 번째 이미지 기준) ---\n"
    txt += (
        f"롤(Roll):  {roll.item():.2f}° (± {rad2deg(results['roll_uncertainty']).item():.2f})°\n"
    )
    txt += (
        f"피치(Pitch): {pitch.item():.2f}° (± {rad2deg(results['pitch_uncertainty']).item():.2f})°\n"
    )
    txt += (
        f"초점 거리(Focal): {camera.f[0, 1].item():.2f} px (± {results['focal_uncertainty'].item():.2f}"
        " px)\n"
    )
    if hasattr(camera, "k1"):
        txt += f"왜곡 계수(K1):    {camera.k1[0].item():.2f}\n"
    return txt


# @spaces.GPU(duration=10)
def inference(img, camera_model):
    out = model.calibrate(img.to(device), camera_model=camera_model)
    save_keys = ["camera", "gravity"] + [
        f"{k}_uncertainty" for k in ["roll", "pitch", "vfov", "focal"]
    ]
    res = {k: v.cpu() for k, v in out.items() if k in save_keys}
    # not converting to numpy results in gpu abort
    res["up_confidence"] = out["up_confidence"].cpu().numpy()
    res["latitude_confidence"] = out["latitude_confidence"].cpu().numpy()
    return res


def process_results(
    image_paths,  # 다중 파일 업로드를 위해 리스트로 변경
    camera_model_korean,  # 한글 카메라 모델명
    plot_up,
    plot_up_confidence,
    plot_latitude,
    plot_latitude_confidence,
    plot_undistort,
):
    """이미지를 처리하고 캘리브레이션 결과를 반환합니다."""

    if not image_paths:
        raise gr.Error("먼저 이미지를 하나 이상 업로드해주세요.")

    # 한글 카메라 모델명을 내부 모델명으로 변환
    internal_camera_model = camera_model_choices[camera_model_korean]

    all_vfovs = []
    first_inference_result = None
    total_calibration_time = 0

    for i, single_image_path in enumerate(image_paths):
        img = model.load_image(single_image_path)
        start = time()
        current_inference_result = inference(img, internal_camera_model)
        total_calibration_time += time() - start
        logger.info(f"캘리브레이션 소요 시간: {time() - start:.2f}초 ({internal_camera_model})")

        if i == 0:  # 첫 번째 이미지 결과를 플로팅 및 기타 파라미터 표시용으로 저장
            first_inference_result = current_inference_result
            first_inference_result["image"] = img.cpu()  # 이미지 텐서 추가

        all_vfovs.append(rad2deg(current_inference_result["camera"].vfov).item())

    avg_vfov = sum(all_vfovs) / len(all_vfovs)

    if first_inference_result is None:  # 이미지 경로가 비어있지 않다면 발생하지 않음
        return ("", np.ones((128, 256, 3)), None)

    # C:/temp/ai_cam_data.txt 파일에 결과 저장
    output_dir = "C:/temp"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "ai_cam_data.txt")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(f"평균 수직 화각 (Average vFoV): {avg_vfov:.2f}°\n")
        f.write(f"--- (아래는 첫 번째 이미지 기준) ---\n")
        f.write(
            f"롤(Roll): {rad2deg(first_inference_result['gravity'].rp).unbind(-1)[0].item():.2f}°\n"
        )
        f.write(
            f"피치(Pitch): {rad2deg(first_inference_result['gravity'].rp).unbind(-1)[1].item():.2f}°\n"
        )
        f.write(f"초점 거리(Focal): {first_inference_result['camera'].f[0, 1].item():.2f} px\n")
        if hasattr(first_inference_result["camera"], "k1"):
            f.write(f"왜곡 계수(K1): {first_inference_result['camera'].k1[0].item():.2f}\n")
    logger.info(f"분석 결과가 {output_file_path}에 저장되었습니다.")

    plot_img = update_plot(
        first_inference_result,  # 플로팅은 첫 번째 이미지 결과로 수행
        plot_up,
        plot_up_confidence,
        plot_latitude,
        plot_latitude_confidence,
        plot_undistort,
    )

    return format_output(first_inference_result, avg_vfov), plot_img, first_inference_result


def update_plot(
    inference_result,
    plot_up,
    plot_up_confidence,
    plot_latitude,
    plot_latitude_confidence,
    plot_undistort,
):
    """Update the plot based on the selected options."""
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
        viz2d.plot_confidences(
            [torch.tensor(inference_result["latitude_confidence"][0])], axes=[ax[0]]
        )

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())

    return img


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            gr.Markdown("""## 입력 이미지""")
            file_input = gr.File(
                label="이미지 시퀀스 업로드 (여러 장 선택 가능)", 
                file_count="multiple", 
                file_types=["image"]
            )
            choice_input = gr.Dropdown(
                choices=list(camera_model_choices.keys()),
                label="카메라 모델을 선택하세요.", value="핀홀 (왜곡 없음 / CG 렌더링 소스용)"
            )
            submit_btn = gr.Button("캘리브레이션 실행 📸")
            gr.Examples(examples=example_images, inputs=[file_input, choice_input])

        with gr.Column():
            gr.Markdown("""## 결과""")
            image_output = gr.Image(label="캘리브레이션 결과")
            gr.Markdown("### 플롯 옵션")
            plot_undistort = gr.Checkbox(
                label="왜곡 보정",
                value=False,
                info="왜곡 보정된 이미지 "
                + "(왜곡 파라미터가 있는 모델에서만 사용 가능하며, "
                + "다른 플롯 옵션을 덮어씁니다).",
            )

            with gr.Row():
                plot_up = gr.Checkbox(label="Up-vector 필드", value=True)
                plot_up_confidence = gr.Checkbox(label="Up-vector 신뢰도", value=False)
                plot_latitude = gr.Checkbox(label="위도(Latitude) 필드", value=True)
                plot_latitude_confidence = gr.Checkbox(label="위도(Latitude) 신뢰도", value=False)

            gr.Markdown("### 캘리브레이션 결과")
            text_output = gr.Textbox(label="추정된 파라미터", type="text", lines=5)

    # Define the action when the button is clicked
    inference_state = gr.State()
    plot_inputs = [
        inference_state,
        plot_up,
        plot_up_confidence,
        plot_latitude,
        plot_latitude_confidence,
        plot_undistort,
    ]
    submit_btn.click(
        fn=process_results,
        inputs=[file_input, choice_input] + plot_inputs[1:],
        outputs=[text_output, image_output, inference_state],
    )

    # Define the action when the plot checkboxes are clicked
    plot_up.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)
    plot_up_confidence.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)
    plot_latitude.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)
    plot_latitude_confidence.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)
    plot_undistort.change(fn=update_plot, inputs=plot_inputs, outputs=image_output)


# Launch the app
demo.launch()
