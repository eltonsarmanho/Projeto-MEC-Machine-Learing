- 1 passo:
criar o método do gráfico que deseja no arquivo machinelearning/prediciont/graphics.py

- 2 passo:
Criar a view no arquivo machinelearning/prediction/views.py
Substituir NomeView pelo nome da view que deseja
Substituir graph_test pelo import do gráfico criado lá no machinelearning/prediction/graphics.py sem renomear a variável context

class NomeView(TemplateView):
    template_name = 'initial.html'
    

    def get_context_data(self, **kwargs):
        context = super(NomeView, self).get_context_data(**kwargs)
        from prediction.graphics import graph_test
        
        context['graph'] = graph_test().to_html()
        return context


- 3 passo:
Criar a url para essa view no machinelearning/machinelearning/urls.py

exemplo:
path('url_desejada/', views.NomeView.as_view()),