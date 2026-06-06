import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from omegaconf import OmegaConf

st.set_page_config(page_title="MARL Reward Landscape", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'default.yml'))
    return OmegaConf.load(config_path)

def main():
    st.title("🐍 Интерактивный ландшафт наград (Plotly Edition)")
    st.markdown("Наведи курсор на любую клетку, чтобы увидеть точную математическую ценность.")

    try:
        config = load_config()
        grid_size = int(config.get('grid_width', 20))
    except Exception as e:
        st.error(f"Не удалось загрузить конфиг: {e}")
        return

    roles = list(config.reward_presets.keys())

    st.sidebar.header("Настройки симуляции")
    selected_role = st.sidebar.selectbox("Выберите роль:", roles)

    presets = config.reward_presets[selected_role]
    
    st.sidebar.markdown("### Награды за события (Sparse)")
    val_food = st.sidebar.slider("Яблоко (food)", 0.0, 300.0, float(presets.get('food', 0.0)), 5.0)
    val_wall = st.sidebar.slider("Стена (wall_penalty)", -100.0, 0.0, float(presets.get('wall_penalty', 0.0)), 1.0)
    val_idle = st.sidebar.slider("Бездействие (idle_penalty)", -5.0, 0.0, float(presets.get('idle_penalty', 0.0)), 0.1)

    st.sidebar.markdown("### Пространственные награды (Dense)")
    val_s_food = st.sidebar.slider("Шаг к еде", -10.0, 10.0, float(presets.get('step_closer_food', 0.0)), 0.5)
    val_s_enemy = st.sidebar.slider("Шаг к врагу", -10.0, 10.0, float(presets.get('step_closer_enemy_harvester', presets.get('step_closer_enemy_hunter', 0.0))), 0.5)
    val_s_ally = st.sidebar.slider("Шаг к союзнику", -10.0, 10.0, float(presets.get('step_closer_team', 0.0)), 0.5)

    st.sidebar.markdown("### Алгоритм RL (MAPPO)")
    gamma = st.sidebar.slider("Дисконт (Gamma γ)", 0.0, 0.99, 0.90, 0.01, help="Как сильно нейросеть 'размазывает' награду от яблока по соседним клеткам.")

    apple_pos = (grid_size // 2 + 2, grid_size - 4)
    enemy_pos = (grid_size - 4, grid_size // 2)
    ally_pos = (4, grid_size // 2 + 3)
    snake_pos = (4, 4)

    Z = np.zeros((grid_size, grid_size))
    max_dist = grid_size * 1.41
    
    for x in range(grid_size):
        for y in range(grid_size):
            if x == 0 or x == grid_size - 1 or y == 0 or y == grid_size - 1:
                Z[y, x] = val_wall * 5
                continue
            
            dist_food = np.hypot(x - apple_pos[0], y - apple_pos[1])
            dist_enemy = np.hypot(x - enemy_pos[0], y - enemy_pos[1])
            dist_ally = np.hypot(x - ally_pos[0], y - ally_pos[1])
            
            val = val_idle
            
            val += val_s_food * (max_dist - dist_food)
            if dist_food == 0:
                val += val_food
            else:
                val += val_food * (gamma ** dist_food)
            
            val += val_s_enemy * (max_dist - dist_enemy)
            if dist_enemy == 0:
                val += 50.0 
            else:
                val += 50.0 * (gamma ** dist_enemy)
            
            if dist_ally > 0: 
                val += val_s_ally * (max_dist - dist_ally)
            
            Z[y, x] = val

    fig = go.Figure(data=go.Heatmap(
        z=Z,
        colorscale='Inferno',
        zmin=-50, 
        zmax=150,
        hoverongaps=False,
        hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z:.2f}<extra></extra>'
    ))

    def add_marker(pos, symbol, color, name, size=16):
        fig.add_trace(go.Scatter(
            x=[pos[0]], y=[pos[1]],
            mode='markers',
            marker=dict(symbol=symbol, size=size, color=color, line=dict(width=2, color='DarkSlateGrey')),
            name=name,
            hovertemplate=f'{name}<extra></extra>'
        ))

    add_marker(apple_pos, 'star', 'yellow', 'Яблоко', 24)
    add_marker(enemy_pos, 'circle', 'red', 'Враг')
    add_marker(ally_pos, 'circle', 'lime', 'Союзник')
    add_marker(snake_pos, 'square', 'cyan', 'Твоя Змейка')

    fig.update_layout(
        width=800,
        height=800,
        xaxis=dict(scaleanchor="y", constrain="domain", showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='black',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()