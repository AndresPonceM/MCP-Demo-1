import os, json, requests, base64, cv2
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mcp.server.fastmcp import FastMCP
from core import opencv_functions  # Importamos tu script exacto

app = FastAPI()
mcp = FastMCP("VisionAgent")
templates = Jinja2Templates(directory="templates")

# Configuración de rutas
SAMPLES_DIR = "image_samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.getenv("HF_TOKEN")

def img_to_base64(img_np):
    """Convierte un array de OpenCV a un string Base64 para HTML"""
    _, buffer = cv2.imencode('.png', img_np)
    return base64.b64encode(buffer).decode('utf-8')

# --- HERRAMIENTA MCP ---
@mcp.tool()
def apply_vision_tool(filename: str, action: str, params_json: str) -> str:
    """Acciones: 'blur' o 'light'. params_json debe ser un JSON con los valores."""
    path = os.path.join(SAMPLES_DIR, filename)
    img = cv2.imread(path)
    params = json.loads(params_json)
    
    if action == "blur":
        res = opencv_functions.desenfocar_imagen(img, params)
    else:
        res = opencv_functions.brillo_contraste_imagen(img, params)
        
    return f"data:image/png;base64,{img_to_base64(res)}"

app.mount("/sse", mcp.get_asgi_app())

# --- RUTAS WEB ---
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
    
    # Prompt Engineering para mapear a tus funciones
    sys_msg = """Responde solo JSON. 
    Si piden desenfoque: {"action": "blur", "params": {"tamaño kernel": 15}} (debe ser impar).
    Si piden brillo/contraste: {"action": "light", "params": {"contraste": 1.2, "brillo": 30}}"""
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": f"<s>[INST] {sys_msg} \n Usuario: {prompt} [/INST]"}
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload).json()
        ai_raw = response[0]['generated_text'].split("[/INST]")[1].strip()
        ai_dec = json.loads(ai_raw)
        
        if ai_dec['action'] == "blur":
            res_img = opencv_functions.desenfocar_imagen(img, ai_dec['params'])
        else:
            res_img = opencv_functions.brillo_contraste_imagen(img, ai_dec['params'])
            
        return {
            "original": fname, 
            "processed_b64": f"data:image/png;base64,{img_to_base64(res_img)}",
            "details": ai_dec
        }
    except:
        return JSONResponse(status_code=500, content={"error": "Error procesando prompt"})