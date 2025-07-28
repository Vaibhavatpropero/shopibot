from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import json

def get_ai_response_2(payload):
    pass

@csrf_exempt
def shopibot_dev(request):
    chat_history = []
    if request.method == "POST":

        # message = request.POST.get("message")
        message = request.body
        print("message:", message)
        message = json.loads(message.decode('utf-8'))
        message = f"{message['message']}\nPlease provide the response in Markdown format with headings, bullet points, and code blocks."
        payload = {
            "query": message,
            "history": chat_history
            }
        # response = get_ai_response_2(payload=payload)
        #chat = Chat(
        #    user=request.user,
        #    message=message,
        #    response=response,
        #    created_at=timezone.now()
        #)
        #chat.save()
        # response = ""
        #chat_history.append({"role": "user", "content": message})
        #chat_history.append({"role": "assistant", "content": response})
        #chat_history = truncate_history(chat_history, 20)
        # return JsonResponse({"message": message, "response": response})
        response = StreamingHttpResponse(get_ai_response_2(payload), status=200, content_type="text/plain")
        print("Response:", response)
        return response
    return render(request, "shopibot-stream-output.html")