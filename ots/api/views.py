from django.http import FileResponse
from rest_framework.views import APIView
from rest_framework.response import Response


class ITS(APIView):

    def post(self, request):
        print(request.data)
        file = request.data['file']
        import os
        print(os.listdir())
        # with open('download.wav', 'rb') as file_handle:
        file_handle = open('download.wav', 'rb')
        response = FileResponse(file_handle, content_type='whatever')
        response['Content-Disposition'] = 'attachment; filename="%s"' % "download.wav"
        return response

        return Response({"message": "You sent a file.",
                         "name": file.name
                         })
