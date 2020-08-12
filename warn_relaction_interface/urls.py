from warn_relaction_interface import view
from django.conf.urls import url
from django.conf import settings
from django.views.static import serve

urlpatterns = [
    url(r'^$', view.start),
    url(r'healthcheck.do', view.start),
    url(r'check_rule', view.check_rule),
    url(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
]
