import plotly.graph_objs as go
import numpy as np

def surface_3d(x_maturities, y_strikes, z_iv, title="Vol Surface (IV)"):
    fig = go.Figure(data=[go.Surface(x=x_maturities, y=y_strikes, z=z_iv)])
    fig.update_layout(title=title, scene=dict(
        xaxis_title='Maturity (years)',
        yaxis_title='Strike',
        zaxis_title='Implied Vol'
    ))
    return fig
