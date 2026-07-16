# --- إعدادات الـ CORS الاحترافية والآمنة لموقعك ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://myaitool.pages.dev",  # رابط موقعك الأساسي على كلاودفلير
        "http://localhost:3000",       # للتطوير المحلي مستقبلاً
        "http://localhost:8000"
    ],
    allow_credentials=True,            # تفعيل استقبال الـ Credentials لمنع حظر المتصفح
    allow_methods=["*"],
    allow_headers=["*"],
)
