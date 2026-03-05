import os, json, base64, cv2
import google.generativeai as genai
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from core import opencv_functions

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Configurar Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

SAMPLES_DIR = "image_samples"
app.mount("/image_samples", StaticFiles(directory=SAMPLES_DIR), name="image_samples")

def img_to_base64(img_np):
    _, buffer = cv2.imencode('.png', img_np)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return templates.TemplateResponse("index.html", {"request": request, "files": files})

@app.post("/ask-ai")
async def ask_ai(data: dict = Body(...)):
    prompt = data.get("prompt")
    fname = data.get("filename")
    path = os.path.join(SAMPLES_DIR, fname)
    img = cv2.imread(path)

    # Instrucciones estrictas para JSON
    sys_msg = (
        "Eres un experto en visión por computadora. Tu tarea es mapear la petición del usuario a una acción de OpenCV.\n"
        "RESPONDE ÚNICAMENTE EN FORMATO JSON.\n"
        "Opciones:\n"
        "1. Si piden desenfoque: {'action': 'blur', 'params': {'tamaño kernel': 15}}\n"
        "2. Si piden luz/brillo/contraste: {'action': 'light', 'params': {'contraste': 1.2, 'brillo': 30}}\n"
        "Petición del usuario: "
    )

    try:
        # Gemini es excelente siguiendo instrucciones de formato
        response = model.generate_content(
            f"{sys_msg} {prompt}",
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json", # Esto fuerza a Gemini a dar JSON puro
            ),
        )
        
        ai_dec = json.loads(response.text)

        if ai_dec['action'] == "blur":
            res_img = opencv_functions.desenfocar_imagen(img, ai_dec['params'])
        else:
            res_img = opencv_functions.brillo_contraste_imagen(img, ai_dec['params'])

        return {
            "original": fname, 
            "processed_b64": f"data:image/png;base64,{img_to_base64(res_img)}",
            "details": ai_dec
        }

    except Exception as e:
        print(f"Error con Gemini: {e}")
        # Fallback local por si acaso falla la cuota gratuita
        res_img = cv2.GaussianBlur(img, (21, 21), 0)
        return {
            "original": fname, 
            "processed_b64": f"data:image/png;base64,{img_to_base64(res_img)}",
            "details": {"action": "blur (emergency)", "info": "Gemini API Limit reached"}
        }