from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt  # Doar pentru testare; vezi comentariul de mai jos
def predict_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        animal = data.get('animal')
        symptoms = data.get('symptoms', [])
        other = data.get('other_symptoms', '')

        # Dummy prediction logic — înlocuiește cu modelul tău ML
        if "blisters" in symptoms:
            predicted_disease = "Foot and Mouth Disease"
        else:
            predicted_disease = "Unknown Disease"

        return JsonResponse({'disease': predicted_disease})
    return JsonResponse({'error': 'Invalid request'}, status=400)