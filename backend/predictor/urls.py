from django.urls import path

from . import views

urlpatterns = [

    path(
        "health/",
        views.health_check,
        name="health"
    ),

    path(
        "predict/",
        views.predict,
        name="predict"
    ),

    path(
        "gradcam/",
        views.gradcam,
        name="gradcam"
    ),

    path(
        "shap/",
        views.shap_analysis,
        name="shap"
    ),

]