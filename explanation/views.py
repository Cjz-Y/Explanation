from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render

from ExplanationServer.settings import LIME_EXPLANATION


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
        if display_type == 1:
            return LIME_EXPLANATION.predict_proba()
        elif display_type == 2:
            return LIME_EXPLANATION.feature_list()
        elif display_type == 3:
            return LIME_EXPLANATION.feature_max()
        elif display_type == 4:
            return LIME_EXPLANATION.feature_value()

        return render(request, 'index.html')

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