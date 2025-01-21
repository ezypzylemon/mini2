from flask import Flask, render_template_string
import mediapipe as mp

app = Flask(__name__)

# HTML 템플릿
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils_3d/control_utils_3d.js" crossorigin="anonymous"></script>
    <title>MediaPipe 3D Control</title>
    <style>
        .landmark_grid_container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f0f0f0;
        }
    </style>
</head>
<body>
    <div class="landmark_grid_container">
        <p>3D Landmark Control will appear here.</p>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

if __name__ == '__main__':
    app.run(debug=True)
