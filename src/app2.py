import gradio as gr

refresh_symbol = '\U0001f504'  # 🔄
switch_values_symbol = '\U000021C5'  # ⇅
camera_symbol = '\U0001F4F7'  # 📷
reverse_symbol = '\U000021C4'  # ⇄
tossup_symbol = '\u2934'
trigger_symbol = '\U0001F4A5'  # 💥
open_symbol = '\U0001F4DD'  # 📝

with gr.Blocks() as webcam_block:
    elem_id_tabname = "webcam"
    tabname = "mediapipe"

    with gr.Row().style(equal_height=True):
        input_image = gr.Image(source='upload', brush_radius=20, mirror_webcam=False, type='numpy')
        cam_image = gr.Image(source="webcam", tool=None)
        # gr.Video(source="webcam")
webcam_block.launch(share=False, debug=True, )
