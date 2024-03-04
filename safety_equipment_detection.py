# Code to apply model on video with 4 classes(Vest, NOVest, Helmet, NOHelmet)

from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


def draw_text_and_box_on_image(image, text, position, box_coordinates, font, color):
    draw = ImageDraw.Draw(image)

    # Draw bounding box
    draw.rectangle(box_coordinates, outline=color, width=2)

    # Draw text with specified color
    draw.text(position, text, fill=color, font=font)

    return image


def process_video(video_path, model, output_folder, output_video_path):
    # Open the video capture
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create VideoWriter object with the same fps as the input video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    frame_number = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Convert the OpenCV BGR image to RGB (PIL format)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        try:
            results = model.predict(image)
            result = results[0]

            # Create a font with the desired size
            font_size = 20  # Adjust the font size as needed
            font = ImageFont.truetype("arial.ttf",
                                      font_size)  # Use arial.ttf or another font file with the desired size

            # Keep track of whether any detection occurred in this frame
            detection_occurred = False

            for idx, box in enumerate(result.boxes):
                class_id = result.names[box.cls[0].item()]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = round(box.conf[0].item(), 2)

                # Determine color based on class_id
                if class_id == "Helmet":
                    bounding_box_color = (0, 255, 0)  # Green
                    text_color = (0, 255, 0)  # Green
                elif class_id == "NOHelmet":
                    bounding_box_color = (0, 0, 255)  # Blue
                    text_color = (0, 0, 255)  # Blue
                elif class_id == "NOVest":
                    bounding_box_color = (255, 0, 0)  # Red
                    text_color = (255, 0, 0)  # Red
                elif class_id == "Vest":
                    bounding_box_color = (255, 255, 0)  # Yellow
                    text_color = (255, 255, 0)  # Yellow
                else:
                    bounding_box_color = (128, 128, 128)  # Default to gray
                    text_color = (128, 128, 128)  # Default to gray

                # Draw text and bounding box on the image
                text = f"{class_id}({conf})"
                position = (cords[0], cords[1] - 22)  # Adjust the position based on your preference
                image_with_text_and_box = draw_text_and_box_on_image(image, text, position, cords, font, text_color)

                # Convert the modified image back to BGR (OpenCV format)
                modified_frame = cv2.cvtColor(np.array(image_with_text_and_box), cv2.COLOR_RGB2BGR)

                # If at least one detection occurred, set detection_occurred to True
                if not detection_occurred:
                    detection_occurred = True

            # Write the frame to the output video (outside the detection loop)
            out.write(modified_frame) if detection_occurred else out.write(frame)

        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")

        frame_number += 1

    # Release the video capture and writer
    cap.release()
    out.release()


# Example usage
output_folder = "Your output folder path"
output_video_path = "Your output video path"
model = YOLO("Your Model path")
video_path = "Your Input video path"
process_video(video_path, model, output_folder, output_video_path)