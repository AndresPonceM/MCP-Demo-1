import os, json, requests, base64, cv2
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mcp.server.fastmcp import FastMCP
from core import opencv_functions

app = FastAPI()
# Nota: Si falla sse_app, prueba cambiar a mcp = FastMCP("VisionAgent", server_transport="sse")
mcp = FastMCP("VisionAgent")
templates = Jinja2Templates(directory="templates")

SAMPLES_DIR = "image_samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

# CORRECCIÓN DE RUTA: Ahora coincide con el HTML
app.mount("/image_samples", StaticFiles(directory=SAMPLES_DIR), name="image_samples")

# Cambiamos a Zephyr o Mistral v0.2 que son más estables en la API gratuita
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_TOKEN = os.getenv("HF_TOKEN")

# PRUEBA DE ARRANQUE: Verás esto en los logs de Railway
print(f"--- DIAGNÓSTICO DE INICIO ---")
print(f"Directorio de imágenes: {os.path.abspath(SAMPLES_DIR)}")
print(f"Archivos encontrados: {os.listdir(SAMPLES_DIR)}")
print(f"Token HF detectado: {'SÍ' if HF_TOKEN else 'NO'}")
print(f"---------------------------")

def img_to_base64(img_np):
    _, buffer = cv2.imencode('.png', img_np)
    return base64.b64encode(buffer).decode('utf-8')

@mcp.tool()
def apply_vision_tool(filename: str, action: str, params_json: str) -> str:
    path = os.path.join(SAMPLES_DIR, filename)
    img = cv2.imread(path)
    params = json.loads(params_json)
    if action == "blur":
        res = opencv_functions.desenfocar_imagen(img, params)
    else:
        res = opencv_functions.brillo_contraste_imagen(img, params)
    return f"data:image/png;base64,{img_to_base64(res)}"

# CORRECCIÓN ATRIBUTO: sse_app para versiones actuales de FastMCP
app.mount("/sse", mcp.sse_app)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return templates.TemplateResponse("index.html", {"request": request, "files": files})

@app.post("/ask-ai")
async def ask_ai(data: dict = Body(...)):
    prompt = data.get("prompt")
    fname = data.get("filename")
    path = os.path.join(SAMPLES_DIR, fname)
    
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": f"Imagen {fname} no encontrada"})

    img = cv2.imread(path)
    sys_msg = """Responde solo JSON puro, sin explicaciones. 
    Si piden desenfoque: {"action": "blur", "params": {"tamaño kernel": 15}}.
    Si piden brillo/contraste: {"action": "light", "params": {"contraste": 1.2, "brillo": 30}}"""
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": f"<s>[INST] {sys_msg} \n Usuario: {prompt} [/INST]"}
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        
        # Extraer JSON de la respuesta del modelo
        gen_text = result[0]['generated_text'] if isinstance(result, list) else result['generated_text']
        ai_raw = gen_text.split("[/INST]")[-1].strip()
        ai_raw = ai_raw.replace("```json", "").replace("```", "").strip()
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
    except Exception as e:
        print(f"Error en ask_ai: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "La IA no respondió correctamente. Revisa el HF_TOKEN."})