from django.shortcuts import render

def home(request):
    return render(request, 'home.html')  # Asumând că e în frontend/templates/home.html