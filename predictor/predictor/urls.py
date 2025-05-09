from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('ai_api.urls')),  
    path('', include('frontend.urls')),  # ← asta trimite "/" către pagina ta HTM
]