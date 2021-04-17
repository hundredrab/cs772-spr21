from django.urls import include, path
from rest_framework import routers
from api import views

router = routers.DefaultRouter()
# router.register(r'its', views.ITS, basename='api')
# router.register(r'groups', views.GroupViewSet)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
    path('api/its/', views.ITS.as_view(), name="its"),
    # path('api/', include('rest_framework.urls', namespace='rest_framework'))
]
