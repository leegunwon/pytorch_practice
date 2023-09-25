import DDPG
import PPO_continueous as PPO
import plotly.graph_objects as go

def main():
    fig = go.figure()
    fig.add_trace(go.Scatter(x=PPO.res,y=range(PPO.res), mode='lines', name='score', color = 'red'))
    fig.add_trace(go.Scatter(x=DDPG.res,y=range(DDPG.res), mode='lines', name='score', color = 'blue'))
    fig.show()