from rest_framework import serializers
# ==========================================================
# Prediction Serializer
# ==========================================================

class PredictionSerializer(serializers.Serializer):

    image = serializers.ImageField(
        required=True
    )

    age = serializers.IntegerField(
        min_value=0,
        max_value=120
    )

    gender = serializers.ChoiceField(
        choices=[
            ("M", "Male"),
            ("F", "Female")
        ]
    )

    view_position = serializers.ChoiceField(
        choices=[
            ("PA", "PA"),
            ("AP", "AP")
        ]
    )


# ==========================================================
# Grad-CAM Serializer
# ==========================================================

class GradCAMSerializer(serializers.Serializer):

    image = serializers.ImageField(
        required=True
    )

    age = serializers.IntegerField(
        min_value=0,
        max_value=120
    )

    gender = serializers.ChoiceField(
        choices=[
            ("M", "Male"),
            ("F", "Female")
        ]
    )

    view_position = serializers.ChoiceField(
        choices=[
            ("PA", "PA"),
            ("AP", "AP")
        ]
    )

    target_class = serializers.CharField()


# ==========================================================
# SHAP Serializer
# ==========================================================

class SHAPSerializer(serializers.Serializer):

    age = serializers.IntegerField(
        min_value=0,
        max_value=120
    )

    gender = serializers.ChoiceField(
        choices=[
            ("M", "Male"),
            ("F", "Female")
        ]
    )

    view_position = serializers.ChoiceField(
        choices=[
            ("PA", "PA"),
            ("AP", "AP")
        ]
    )