# -*- coding: utf-8 -*-
"""Filter.py"""

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import cv2, numpy as np
from PIL import Image
import io, base64

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# ---------------- Basic Filters ---------------- #
def apply_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def apply_invert(img): return cv2.bitwise_not(img)
def apply_brightness(img, value: int = 30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
def apply_contrast(img, alpha: float = 1.3): return cv2.convertScaleAbs(img, alpha=alpha, beta=0)
def apply_sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)
def apply_blur(img, ksize: int = 15): return cv2.GaussianBlur(img, (ksize, ksize), 0)
def apply_edges(img, t1=100, t2=200): return cv2.Canny(img, t1, t2)
def apply_emboss(img):
    kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
    return cv2.filter2D(img, -1, kernel)
def apply_sketch(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv=255-gray
    blur=cv2.GaussianBlur(inv,(21,21),0)
    return cv2.divide(gray,255-blur,scale=256)

# ---------------- Color / Artistic ---------------- #
def apply_sepia(img):
    kernel = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
    return cv2.transform(img, kernel)
def apply_warm(img, strength=25): return cv2.addWeighted(img, 1.2, np.zeros(img.shape,img.dtype), 0, strength)
def apply_cool(img, strength=10): return cv2.addWeighted(img, 0.8, np.zeros(img.shape,img.dtype), 0, strength)
def apply_hdr(img): return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
def apply_vintage(img):
    img = img.astype(np.float32)
    rows, cols = img.shape[:2]
    kernel = cv2.getGaussianKernel(cols, 200) * cv2.getGaussianKernel(rows, 200).T
    mask = 255 * kernel / np.linalg.norm(kernel)
    for i in range(3):
        img[:, :, i] *= mask
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
def apply_cartoon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)
def apply_watercolor(img): return cv2.stylization(img, sigma_s=60, sigma_r=0.6)
def apply_pencil(img): return cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07)[0]
def apply_oil(img): return cv2.xphoto.oilPainting(img,7,1)

# ---------------- Unique / New Filters ---------------- #
def apply_solarize(img, threshold=128):
    img = img.copy()
    img[img > threshold] = 255 - img[img > threshold]
    return img

def apply_posterize(img, levels=4):
    factor = 256 // levels
    return (img // factor) * factor

def apply_vignette(img, intensity=0.8):
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / np.max(kernel)
    mask = 1 - intensity * (1 - mask)
    output = np.empty_like(img, dtype=np.float32)
    for i in range(3):
        output[:,:,i] = img[:,:,i] * mask
    return np.clip(output, 0, 255).astype(np.uint8)

def apply_edge_enhance(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def apply_hue_shift(img, shift=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = (hsv[:,:,0] + shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_saturation_boost(img, factor=1.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_invert_blue(img):
    img = img.copy()
    img[:,:,0] = 255 - img[:,:,0]
    return img

def apply_red_boost(img, factor=1.3):
    b, g, r = cv2.split(img)
    r = np.clip(r * factor, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

def apply_green_boost(img, factor=1.3):
    b, g, r = cv2.split(img)
    g = np.clip(g * factor, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

def apply_blue_boost(img, factor=1.3):
    b, g, r = cv2.split(img)
    b = np.clip(b * factor, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

def apply_chromatic_aberration(img, shift=3):
    b, g, r = cv2.split(img)
    r_shifted = np.roll(r, shift, axis=1)
    b_shifted = np.roll(b, -shift, axis=1)
    return cv2.merge([b_shifted, g, r_shifted])

def apply_pixelate(img, pixel_size=8):
    h, w = img.shape[:2]
    temp = cv2.resize(img, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_halftone(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    halftone = np.zeros_like(img)
    radius = 3
    for y in range(0, gray.shape[0], radius*2):
        for x in range(0, gray.shape[1], radius*2):
            intensity = gray[y:y+radius*2, x:x+radius*2].mean() / 255
            circle_radius = int(radius * (1 - intensity))
            if circle_radius > 0:
                cv2.circle(halftone, (x+radius, y+radius), circle_radius, (255, 255, 255), -1)
    return halftone

def apply_thermal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def apply_night_vision(img):
    green_tint = np.zeros_like(img)
    green_tint[:,:,1] = 255
    return cv2.addWeighted(img, 0.7, green_tint, 0.3, 0)

def apply_cyanotype(img):
    blue_filter = np.array([[[0.7, 0.5, 0.2]]], dtype=np.float32)
    cyanotype = cv2.transform(img, blue_filter)
    return np.clip(cyanotype, 0, 255).astype(np.uint8)

def apply_gotham(img):
    b, g, r = cv2.split(img)
    r = np.clip(r * 1.1, 0, 255).astype(np.uint8)
    g = np.clip(g * 0.9, 0, 255).astype(np.uint8)
    b = np.clip(b * 1.2, 0, 255).astype(np.uint8)
    filtered = cv2.merge([b, g, r])
    return apply_contrast(filtered, 1.2)

def apply_clarity(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

def apply_dramatic_bw(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def apply_foggy(img, intensity=0.3):
    fog = np.full_like(img, 200)
    return cv2.addWeighted(img, 1-intensity, fog, intensity, 0)

def apply_golden_hour(img):
    golden_tint = np.array([[[15, 75, 150]]], dtype=np.float32)
    golden = cv2.transform(img, golden_tint)
    return np.clip(golden, 0, 255).astype(np.uint8)

# ---------------- LUT / Color Shifts ---------------- #
def color_shift(img, r_shift=0,g_shift=0,b_shift=0):
    b,g,r=cv2.split(img)
    r=cv2.add(r,r_shift); g=cv2.add(g,g_shift); b=cv2.add(b,b_shift)
    return cv2.merge([b,g,r])

def lut_filter(img,lut): return cv2.LUT(img,lut)
def build_lut(gamma=1.0):
    invGamma = 1.0/gamma
    return np.array([((i/255.0)**invGamma)*255 for i in np.arange(256)]).astype("uint8")

FILTERS = {
    "Gray": apply_gray, "Invert": apply_invert, "Brightness": apply_brightness, "Contrast": apply_contrast,
    "Sharpen": apply_sharpen, "Blur": apply_blur, "Edges": apply_edges, "Emboss": apply_emboss, "Sketch": apply_sketch,
    "Sepia": apply_sepia, "Warm": apply_warm, "Cool": apply_cool, "HDR": apply_hdr, "Vintage": apply_vintage,
    "Cartoon": apply_cartoon, "Watercolor": apply_watercolor, "Pencil": apply_pencil, "Oil": apply_oil,
    "Solarize": apply_solarize, "Posterize": apply_posterize, "Vignette": apply_vignette,
    "Edge Enhance": apply_edge_enhance, "Hue Shift": apply_hue_shift,
    "Saturation": apply_saturation_boost, "Invert Blue": apply_invert_blue,
    "Red Boost": apply_red_boost, "Green Boost": apply_green_boost, "Blue Boost": apply_blue_boost,
    "Chromatic Aberration": apply_chromatic_aberration, "Pixelate": apply_pixelate,
    "Halftone": apply_halftone, "Thermal": apply_thermal, "Night Vision": apply_night_vision,
    "Cyanotype": apply_cyanotype, "Gotham": apply_gotham, "Clarity": apply_clarity,
    "Dramatic_bw": apply_dramatic_bw, "Foggy": apply_foggy, "Golden Hour": apply_golden_hour,
}

# ------------------- Helpers ------------------- #
def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")

# ------------------- Single Filter API ------------------- #
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    options = "".join([f"<option value='{f}'>{f}</option>" for f in FILTERS.keys()])
    return f"""
    <html>
    <head><title>Single Filter</title></head>
    <body>
    <h2>🎨 Apply Single Filter</h2>
    <form action="/apply_filter/" method="post" enctype="multipart/form-data">
        <label>Select Filter:</label>
        <select name="filter_name">{options}</select><br><br>
        <label>Adjustment (optional):</label>
        <input type="text" name="adjust_value" placeholder="30 or 1.5"><br><br>
        <input type="file" name="file" accept="image/*" required><br><br>
        <button type="submit">Apply Filter</button>
    </form>
    <hr>
    <a href="/multi_filter_auto/">Switch to Multi Filter API</a>
    </body>
    </html>
    """

@app.post("/apply_filter/", response_class=HTMLResponse)
async def apply_filter(filter_name: str = Form(...), adjust_value: str = Form(None), file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    func = FILTERS.get(filter_name)
    if not func:
        return HTMLResponse(f"<h2>❌ Invalid filter: {filter_name}</h2>")

    try:
        if adjust_value:
            val = float(adjust_value) if '.' in adjust_value else int(adjust_value)
            filtered = func(img, val)
        else:
            filtered = func(img)
    except TypeError:
        filtered = func(img)

    encoded_img = encode_image(filtered)
    return f"""
    <html>
    <head><title>Filtered Result</title></head>
    <body>
        <h2>✅ Filter Applied: {filter_name}</h2>
        <img src="data:image/jpeg;base64,{encoded_img}" style="max-width:500px;"><br><br>
        <a href="/">🔙 Apply Another</a>
    </body>
    </html>
    """

# ------------------- Multi Filter API ------------------- #
@app.get("/multi_filter_auto/", response_class=HTMLResponse)
async def multi_filter_auto_home():
    return f"""
    <html>
    <head><title>Auto Multi Filter</title></head>
    <body>
    <h2>🎨 Applying All Filters Automatically</h2>
    <form action="/apply_multi_filter_auto/" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required><br><br>
        <button type="submit">Apply All Filters</button>
    </form>
    <hr>
    <a href="/">Switch to Single Filter API</a>
    </body>
    </html>
    """

@app.post("/apply_multi_filter_auto/", response_class=HTMLResponse)
async def apply_multi_filter_auto(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    result_html = ""

    for filter_name, func in FILTERS.items():
        try:
            filtered = func(img.copy())
            encoded_img = encode_image(filtered)
            result_html += f"""
            <div style="display:inline-block;text-align:center;margin:10px;">
                <h4>{filter_name}</h4>
                <img src='data:image/jpeg;base64,{encoded_img}' style='max-width:200px;'>
            </div>
            """
        except Exception as e:
            print(f"Skipping {filter_name} due to error: {e}")
            continue

    return f"""
    <html>
    <head><title>All Filters Result</title></head>
    <body>
        <h2>✅ All Filters Applied (Auto)</h2>
        {result_html}
        <br><br>
        <a href="/multi_filter_auto/">🔙 Try Another Image</a>
    </body>
    </html>
    """

@app.post("/apply_multi_filter_auto_json/")
async def apply_multi_filter_auto_json(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    results = []
    for filter_name, func in FILTERS.items():
        try:
            filtered = func(img.copy())
            encoded_img = encode_image(filtered)
            results.append({"filter_name": filter_name, "image_base64": encoded_img})
        except Exception as e:
            print(f"Skipping {filter_name} due to error: {e}")
            continue
    return JSONResponse(content={"filters_applied": results})

# ------------------- Run ------------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
