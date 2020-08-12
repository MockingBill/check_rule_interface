from check_rule import view
from django.conf.urls import url
from django.conf import settings
from django.views.static import serve

urlpatterns = [
    url(r'^$', view.start),
    url(r'healthcheck.do', view.start),
    url(r'warn_realtion', view.warn_realtion),
    url(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
]
