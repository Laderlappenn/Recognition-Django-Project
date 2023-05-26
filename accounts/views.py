from django.shortcuts import render, HttpResponseRedirect
from .forms import RegistrationForm
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from .models import Account
from django.shortcuts import get_object_or_404
from django.core.paginator import Paginator

@login_required
def profile(request):
    user_id = request.user.id
    queryset = Account.objects.get(id=user_id)  # get_object_or_404(Account, pk=user_id)
    return render(request, 'main/profile.html', {'profile': queryset})


# def register(request):
#     if request.method == 'GET':
#         form = RegistrationForm()
#         return render(request, 'accounts/register.html', {'form': form})
#
#     elif request.method == 'POST':
#         form = RegistrationForm(request.POST)
#         if form.is_valid():
#             form.save()
#             username = form.cleaned_data['username']
#             password = form.cleaned_data['password1']
#             user = authenticate(username=username, password=password)
#             login(request, user)
#             messages.success(request, ('успешная регистрация'))
#             return HttpResponseRedirect('../../')
#         else:
#             return render(request, 'accounts/register.html', {'form': form})
#

class BBLoginView(LoginView):
    template_name = 'main/login.html'


class BBLogoutView(LoginRequiredMixin, LogoutView):
    template_name = 'accounts/logout.html'



# @login_required
# def users(request):
#     if request.user.type == 'DISPATCHER':
#         queryset = Account.objects.all().order_by('-date_updated')
#         paginator = Paginator(queryset, 10)
#         page_number = request.GET.get('page')
#         page_obj = paginator.get_page(page_number)
#         return render(request, 'accounts/users.html', {"page_obj": page_obj})
#
# @login_required
# def user(request, pkey):
#     if request.user.type == 'DISPATCHER':
#         queryset = Table_1.objects.select_related('user').filter(user_id=pkey).order_by('date_updated')
#         profile = Account.objects.get(id=pkey)
#         username = profile.username
#         paginator = Paginator(queryset, 10)
#         page_number = request.GET.get('page')
#         page_obj = paginator.get_page(page_number)
#         return render(request, 'acts/tables.html', {"page_obj": page_obj, "pkey": pkey, "username": username})