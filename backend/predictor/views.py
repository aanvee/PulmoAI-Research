from django.shortcuts import render

# Create your views here.
import os
import tempfile

from django.http import JsonResponse
from rest_framework.decorators import api_view

from .serializers import (
    PredictionSerializer,
    GradCAMSerializer,
    SHAPSerializer
)

from .services import (
    predict_service,
    gradcam_service,
    shap_service
)


# ==========================================================
# Health Check
# ==========================================================

@api_view(["GET"])
def health_check(request):

    return JsonResponse({

        "status": "healthy",

        "message": "PulmoAI Backend Running"

    })


# ==========================================================
# Prediction API
# ==========================================================

@api_view(["POST"])
def predict(request):

    serializer = PredictionSerializer(data=request.data)

    if not serializer.is_valid():

        return JsonResponse(
            serializer.errors,
            status=400
        )

    image = serializer.validated_data["image"]

    age = serializer.validated_data["age"]

    gender = serializer.validated_data["gender"]

    view_position = serializer.validated_data["view_position"]

    temp_dir = tempfile.gettempdir()

    image_path = os.path.join(
        temp_dir,
        image.name
    )

    with open(image_path, "wb+") as destination:

        for chunk in image.chunks():

            destination.write(chunk)

    predictions = predict_service(

        image_path,

        age,

        gender,

        view_position

    )

    return JsonResponse({

        "success": True,

        "predictions": predictions

    })


# ==========================================================
# GradCAM API
# ==========================================================

@api_view(["POST"])
def gradcam(request):

    serializer = GradCAMSerializer(data=request.data)

    if not serializer.is_valid():

        return JsonResponse(
            serializer.errors,
            status=400
        )

    image = serializer.validated_data["image"]

    age = serializer.validated_data["age"]

    gender = serializer.validated_data["gender"]

    view_position = serializer.validated_data["view_position"]

    target_class = serializer.validated_data["target_class"]

    temp_dir = tempfile.gettempdir()

    image_path = os.path.join(
        temp_dir,
        image.name
    )

    with open(image_path, "wb+") as destination:

        for chunk in image.chunks():

            destination.write(chunk)

    gradcam_image = gradcam_service(

        image_path,

        age,

        gender,

        view_position,

        target_class

    )

    return JsonResponse({

        "success": True,

        "gradcam_image": gradcam_image

    })


# ==========================================================
# SHAP API
# ==========================================================

@api_view(["POST"])
def shap_analysis(request):

    serializer = SHAPSerializer(data=request.data)

    if not serializer.is_valid():

        return JsonResponse(
            serializer.errors,
            status=400
        )

    age = serializer.validated_data["age"]

    gender = serializer.validated_data["gender"]

    view_position = serializer.validated_data["view_position"]

    shap_plot = shap_service(

        age,

        gender,

        view_position

    )

    return JsonResponse({

        "success": True,

        "shap_plot": shap_plot

    })