"""
╔══════════════════════════════════════════════════════════════╗
║         SIMULADOR — ALGORITMO DE DIJKSTRA                    ║
║         Práctica 3 — Inteligencia Artificial                 ║
║         CETI Colomos  |  Grupo 6E  |  23310384               ║
╚══════════════════════════════════════════════════════════════╝

"""

import heapq
import matplotlib
# ─── [CODESPACE] Si corres en GitHub Codespace, descomenta la siguiente línea:
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ─── GRAFO DE EJEMPLO ────────────────────────────────────────────────────────
GRAPH = {
    'A': [('B', 4),  ('C', 2)],
    'B': [('A', 4),  ('C', 5),  ('D', 10)],
    'C': [('A', 2),  ('B', 5),  ('E', 3)],
    'D': [('B', 10), ('E', 4),  ('F', 11)],
    'E': [('C', 3),  ('D', 4),  ('F', 7)],
    'F': [('D', 11), ('E', 7)],
}

START_NODE  = 'A'
END_NODE    = 'F'
PAUSE_TIME  = 1.2    # segundos entre pasos (ajústalo a tu gusto)
SAVE_GIF    = False  # True → también guarda dijkstra_animacion.gif
GIF_OUTPUT  = 'dijkstra_animacion.gif'

# ─── COLORES ─────────────────────────────────────────────────────────────────
C_UNVISITED = '#AED6F1'
C_INQUEUE   = '#F9E79F'
C_CURRENT   = '#F0A500'
C_VISITED   = '#58D68D'
C_EDGE_DEF  = '#BDC3C7'
C_EDGE_VIS  = '#27AE60'
C_EDGE_PATH = '#E74C3C'

SEP  = '─' * 58
SEP2 = '═' * 58

# ─── CONSOLA ─────────────────────────────────────────────────────────────────

def print_state(step, distances, visited, current, in_queue, action=''):
    print(f'\n{SEP}')
    print(f'  PASO {step}  |  {action}')
    print(SEP)
    print(f'  {"Nodo":<8} {"Distancia":<12} {"Estado"}')
    print(f'  {"----":<8} {"---------":<12} {"------"}')
    for node in sorted(distances):
        d     = distances[node]
        d_str = str(d) if d != float('inf') else 'inf'
        if node == current:
            estado = '<-- PROCESANDO AHORA'
        elif node in visited:
            estado = '[ok] Visitado (definitivo)'
        elif node in in_queue:
            estado = '[..] En cola de prioridad'
        else:
            estado = '[ ] Sin explorar'
        print(f'  {node:<8} {d_str:<12} {estado}')
    print(SEP)

# ─── GRAFO NETWORKX ──────────────────────────────────────────────────────────

def build_nx_graph(graph):
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors:
            G.add_edge(node, neighbor, weight=weight)
    return G

# ─── DIBUJO ──────────────────────────────────────────────────────────────────

def draw(G, pos, ax, fig, distances, visited, current, in_queue,
         visited_edges, path_edges, title, frames=None):
    ax.clear()

    node_colors = []
    for node in G.nodes():
        if node == current:        node_colors.append(C_CURRENT)
        elif node in visited:      node_colors.append(C_VISITED)
        elif node in in_queue:     node_colors.append(C_INQUEUE)
        else:                      node_colors.append(C_UNVISITED)

    edge_colors, edge_widths = [], []
    for u, v in G.edges():
        if (u,v) in path_edges or (v,u) in path_edges:
            edge_colors.append(C_EDGE_PATH); edge_widths.append(4.5)
        elif (u,v) in visited_edges or (v,u) in visited_edges:
            edge_colors.append(C_EDGE_VIS);  edge_widths.append(2.5)
        else:
            edge_colors.append(C_EDGE_DEF);  edge_widths.append(1.5)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=1000, linewidths=2,
                           edgecolors='#2C3E50')

    labels = {n: f'{n}\n({"inf" if distances.get(n)==float("inf") else distances.get(n, "?")})'
              for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=9, font_weight='bold', font_color='#1A252F')
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'),
        ax=ax, font_size=8, font_color='#555',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    leyenda = [
        mpatches.Patch(color=C_UNVISITED, label='Sin explorar'),
        mpatches.Patch(color=C_INQUEUE,   label='En cola de prioridad'),
        mpatches.Patch(color=C_CURRENT,   label='Procesando (actual)'),
        mpatches.Patch(color=C_VISITED,   label='Visitado (definitivo)'),
    ]
    if path_edges:
        leyenda.append(mpatches.Patch(color=C_EDGE_PATH, label='Camino minimo'))
    ax.legend(handles=leyenda, loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=12)
    ax.axis('off')
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Capturar frame para GIF si está activado
    if frames is not None:
        from PIL import Image
        import numpy as np
        buf = fig.canvas.buffer_rgba()
        img = Image.frombuffer('RGBA', fig.canvas.get_width_height(),
                               buf, 'raw', 'RGBA', 0, 1)
        frames.append(img.convert('RGB'))

# ─── DIJKSTRA ─────────────────────────────────────────────────────────────────

def reconstruct_path(previous, start, end):
    path, c = [], end
    while c:
        path.append(c); c = previous.get(c)
    path.reverse()
    return path if path and path[0] == start else []

def path_to_edges(path):
    return {(path[i], path[i+1]) for i in range(len(path)-1)}

def run_dijkstra(graph, start, end, G, pos, ax, fig):
    distances     = {n: float('inf') for n in graph}
    distances[start] = 0
    previous      = {n: None for n in graph}
    pq            = [(0, start)]
    visited       = set()
    in_queue      = {start}
    visited_edges = set()
    frames        = [] if SAVE_GIF else None
    step          = 0

    print(f'\n{SEP2}')
    print(f'  SIMULADOR DE DIJKSTRA  |  CETI Colomos 6E  |  23310384')
    print(f'  Origen: {start}    Destino: {end}')
    print(SEP2)

    print_state(step, distances, visited, None, in_queue, 'Inicializacion')
    draw(G, pos, ax, fig, distances, visited, None, in_queue,
         set(), set(), f'Paso 0 - Inicializacion\n{start}=0, resto=inf',
         frames=frames)
    plt.pause(PAUSE_TIME)

    while pq:
        curr_dist, curr = heapq.heappop(pq)
        if curr in visited:
            continue

        step += 1
        visited.add(curr)
        in_queue.discard(curr)

        print_state(step, distances, visited, curr, in_queue,
                    f'Extrayendo nodo {curr} | dist={curr_dist}')
        draw(G, pos, ax, fig, distances, visited, curr, in_queue,
             visited_edges, set(),
             f'Paso {step} - Procesando nodo {curr}\n'
             f'Distancia minima confirmada: {curr_dist}',
             frames=frames)
        plt.pause(PAUSE_TIME)

        if curr == end:
            break

        for neighbor, weight in graph[curr]:
            if neighbor in visited:
                continue
            new_dist = curr_dist + weight
            step    += 1

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor]  = curr
                heapq.heappush(pq, (new_dist, neighbor))
                in_queue.add(neighbor)
                visited_edges.add((curr, neighbor))
                nota = f'ACTUALIZADO a {new_dist}'
            else:
                nota = f'sin cambio (actual={distances[neighbor]} <= {new_dist})'

            print(f'  -> Relajando {curr}->{neighbor}  peso={weight}  {nota}')
            draw(G, pos, ax, fig, distances, visited, curr, in_queue,
                 visited_edges, set(),
                 f'Paso {step} - Relajando {curr} -> {neighbor}\n'
                 f'{curr_dist} + {weight} = {new_dist} | {nota}',
                 frames=frames)
            plt.pause(PAUSE_TIME * 0.7)

    # Resultado final
    path       = reconstruct_path(previous, start, end)
    path_edges = path_to_edges(path)
    total      = distances[end]

    print(f'\n{SEP2}')
    print(f'  RESULTADO FINAL')
    print(SEP2)
    print(f'  Camino minimo: {" -> ".join(path)}')
    print(f'  Distancia total: {total}')
    print(f'\n  Tabla de distancias desde {start}:')
    for n in sorted(distances):
        d    = distances[n]
        pred = previous[n] or '--'
        print(f'    {n}  dist={str(d):<6}  predecesor={pred}')
    print(SEP2)

    draw(G, pos, ax, fig, distances, visited, None, set(),
         visited_edges, path_edges,
         f'RESULTADO: {" -> ".join(path)}  |  Distancia total: {total}',
         frames=frames)

    if SAVE_GIF and frames:
        frames[0].save(GIF_OUTPUT, save_all=True,
                       append_images=frames[1:], duration=1100, loop=0)
        print(f'\n  GIF guardado: {GIF_OUTPUT}')

    plt.pause(0.1)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    G   = build_nx_graph(GRAPH)
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#F0F3F4')
    ax.set_facecolor('#F8F9FA')

    plt.ion()
    plt.show()

    run_dijkstra(GRAPH, START_NODE, END_NODE, G, pos, ax, fig)

    print('\n  Cierra la ventana para terminar.')
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
