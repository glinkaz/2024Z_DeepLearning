import plotly.graph_objs as go
from plotly.subplots import make_subplots


def plot_batch_training(dfs, trace_names, colors):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy Plot", "Validation Accuracy Plot"))
    for i, (df, trace_name) in enumerate(zip(dfs, trace_names)):
        color = colors[i]
        fig.add_trace(
            go.Scatter(x=df['Epoch'], y=df[' Accuracy'], mode='lines', name=trace_name, line=dict(color=color)), row=1,
            col=1)
        fig.add_trace(go.Scatter(x=df['Epoch'], y=df[' Accuracy_valid'], mode='lines', name=f'Gaussian Noise 0.{i + 1}',
                                 showlegend=False, line=dict(color=color)), row=1, col=2)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Validation Accuracy", row=1, col=2)

    max_y = max([df[[' Accuracy', ' Accuracy_valid']].max().max() for df in dfs])
    min_y = max([df[[' Accuracy', ' Accuracy_valid']].min().min() for df in dfs])
    fig.update_yaxes(range=[min_y, max_y], row=1, col=1)
    fig.update_yaxes(range=[min_y, max_y], row=1, col=2)

    fig.show()