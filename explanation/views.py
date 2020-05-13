from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render


def index(request):
    return render(request, 'index.html')


def lime_demo(request):
    """
    POST请求，根据submit的值来进行不同的展示
    :param request:
    :return:
    """
    if request.method == 'GET':
        return render(request, 'index.html')
    else:
        display_type = int(request.POST.get('display_type', '-1'))
        if display_type == -1:
            pass
        elif display_type == 1:
            pass
        elif display_type == 2:
            pass
        else:
            pass

        # try:
        #     display_type = request.POST['display_type']
        # except:
        #     pass
        # else:
        #     pass


def say_hi(request):
    return HttpResponse('Hello World')


def test_html(request):
    return render(request, 'first_test.html')