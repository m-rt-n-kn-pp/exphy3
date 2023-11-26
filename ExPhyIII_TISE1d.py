import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import eigh_tridiagonal
from sympy import symbols, Symbol, lambdify, parse_expr, latex


# COMPUTE EIGENSTATES & ANIMATED PLOT


def eigenstates(potential, N=500):
    """
    Lösung für ein räumlich beschränktes Teilchen.

    Parameters:
    -----------
    potential : callable
        Potential V(x) innerhalb eines unendlichen Potentialkasten für x in [0,1].
    N : int, optional (default=500)
        Anzahl der Diskretisierungspunkte.

    Returns:
    --------
    x : numpy.ndarray, shape (N,)
        Positionen.
    E_n : numpy.ndarray, shape (N-2,)
        Energieeigenwerte.
    phi_n : numpy.ndarray, shape (N-2, N)
        Eigenzustände phi_n(x) in Zeilen.
    """
    N -= 1
    dx = 1 / N
    x = np.linspace(0, 1, N + 1)
    d = 1 / dx**2 + np.vectorize(potential)(x[1:-1])  # diagonal
    e = np.full(len(d) - 1, -1 / (2 * dx**2))  # off-diagonal
    E_n, phi_n = eigh_tridiagonal(d, e)
    phi_n = np.pad(phi_n.T, ((0, 0), (1, 1)))  # include boundary values
    norm = np.sum(np.abs(phi_n) ** 2, axis=1) * dx  # normalize rows
    phi_n /= norm[:, np.newaxis] ** 0.5
    return x, E_n, phi_n


def eigenstates_animation(
    x,
    potential,
    E_n,
    phi_n,
    n_max=10,
    use_box_border=True,
    show_trail=True,
    potential_latex_expr=None,
):
    # TODO: phi_n complex? -> plot Re, Im, abs + limit ylims different
    # but bound, confined eigenstates seem to be real
    E_n = E_n[: n_max + 1]
    N = len(E_n)
    phi_n = phi_n[: n_max + 1, :]
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])  # fig = go.Figure()

    # Potential plot, blue: rgb(37, 53, 68), green: rgb(89, 117, 109)
    V = np.vectorize(potential)(x)
    fig.add_trace(
        go.Scatter(
            visible=True,
            name=r"$\large{\mathrm{Potential}}$",
            x=x,
            y=V,
            line=dict(color="rgb(37, 53, 68)"),
        ),
        secondary_y=True,
    )

    # Potential well border plot: arrows or filled box
    border_box = [min(-1, np.min(V)), max(1, np.max(V))]
    if not use_box_border:
        arrow_length = border_box[1]
        arrows = []
        for arrow_x in [0, 1]:
            arrow = go.layout.Annotation(
                x=arrow_x,
                y=arrow_length,
                xref="x",
                yref="y2",
                ax=arrow_x,
                ay=0,
                axref="x",
                ayref="y2",
                text="",
                yanchor="top",
                showarrow=True,
                arrowhead=2,
                arrowwidth=2,
                arrowcolor="rgb(37, 53, 68)",
            )
            arrows.append(arrow)
        fig.update_layout(annotations=arrows)
    if use_box_border:
        for box_x in [(-0.02, 0), (1, 1.02)]:
            fig.add_shape(
                type="rect",
                x0=box_x[0],
                y0=border_box[0],
                x1=box_x[1],
                y1=border_box[1],
                fillcolor="rgb(37, 53, 68)",
                line=dict(width=2, color="rgb(37, 53, 68)"),
                xref="x",
                yref="y2",
            )

    # Eigenstates plot, orange: rgb(245, 110, 73), brightgreen: rgb(17, 137, 113)
    # Add traces and steps for slider
    steps = []
    for i, E_i in enumerate(E_n):
        # Add animated plot corresponding to slider
        visible_beginning = i == 0  # first one visible even before startin animation
        fig.add_trace(
            go.Scatter(
                visible=visible_beginning,
                name=r"$\large{\mathrm{Wellenfunktion}}$",  # rf'$\text{{Wellenfunktion n={i:0{padding_len}d}}}$'
                x=x,
                y=phi_n[i, :],
                line=dict(color="rgb(17, 137, 113)"),
            )
        )

        if show_trail:
            # Add color alpha trail plot of all Eigenstates, TODO: only plot j<i
            for j, phi in enumerate(phi_n):
                alpha = 0.5 * np.exp(-0.8 * abs(j - i)) * (j < i)
                alpha *= alpha > 0.01
                fig.add_trace(
                    go.Scatter(
                        visible=False,
                        showlegend=False,
                        x=x,
                        y=phi,
                        line=dict(color=f"rgba(17, 137, 113, {alpha})"),
                    )
                )

        # Visibility of traces depending depending on slider position:
        if show_trail:
            # True at slider position + trail (together length N+1) + first trace is always visible potential plot
            visible = (
                [True]
                + [False] * (N + 1) * i
                + [True] * (N + 1)
                + [False] * (N + 1) * (N - i - 1)
            )
        else:
            visible = [True] + [False] * N
            visible[i + 1] = True
        step = dict(
            method="update",
            args=[
                {"visible": visible},
                {
                    "title": rf"$\text{{Eigenzustand zu }}E_{{{i}}} = {E_i:.2f} \frac{{\hbar^2}}{{m L^2}}.$"
                },
            ],
            label=str(i),
        )
        steps.append(step)

    # Create slider
    slider = go.layout.Slider(
        currentvalue={"prefix": "n = "},
        pad={"t": 50},
        steps=steps,
        len=0.95,
        x=0,
        xanchor="left",
    )

    title = "Eigenzustände für ein gegebenes Potential."
    if potential_latex_expr:
        title = rf"$\text{{Eigenzustände für das Potential }}{potential_latex_expr}.$"

    fig.update_layout(
        sliders=[slider],
        title=title,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.94),
    )

    phi_range = [np.min(phi_n), np.max(phi_n)]
    padding = (phi_range[1] - phi_range[0]) * 0.05
    phi_range[0] -= padding
    phi_range[1] += padding

    padding_potential = (border_box[1] - border_box[0]) * 0.05
    border_box[0] -= padding_potential
    border_box[1] += padding_potential
    
    #for i, f in enumerate(fig.frames[1:]):  # first frame is potential
    #    E_i = E_n[i]
    #    if E_i > 0:
    #        border_box_adjust = [border_box[0], E_i + border_box[1]]
    #    else:
    #        border_box_adjust = [E_i + border_box[0], border_box[1]]
    #    f.layout.update(yaxis=dict(range=border_box_adjust, secondary_y=True))

    fig.update_yaxes(
        title_text=r"$\large{\phi_n}$",
        range=phi_range,
        showgrid=False,
        title_font=dict(color="rgb(17, 137, 113)"),
        tickfont=dict(color="rgb(17, 137, 113)"),
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=r"$\large{V \cdot m L^2\hbar^{-2}}$",
        range=border_box,
        showgrid=False,
        title_font=dict(color="rgb(37, 53, 68)"),
        tickfont=dict(color="rgb(37, 53, 68)"),
        secondary_y=True,
    )
    fig.update_xaxes(title_text=r"$\large{x / L}$")

    return fig


# DASH APP - POTENTIAL INPUT FIELD

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])  # BOOTSTRAP, DARKLY, FLATLY

dropdown_menu_items = [
    dbc.DropdownMenuItem('Step', id="dropdown-menu-item-1"),
    dbc.DropdownMenuItem("Barrier", id="dropdown-menu-item-2"),
    dbc.DropdownMenuItem("Double Barrier", id="dropdown-menu-item-3"),
    dbc.DropdownMenuItem("Multi Potential Well", id="dropdown-menu-item-4"),
    dbc.DropdownMenuItem("Gauß", id="dropdown-menu-item-5"),
    dbc.DropdownMenuItem("Sine", id="dropdown-menu-item-6"),
    dbc.DropdownMenuItem(divider=True),
    dbc.DropdownMenuItem("Infinite square well", id="dropdown-menu-item-clear"),
]

func_examples = {
    'dropdown-menu-item-1' : '1000 * Heaviside(x-0.5)',
    'dropdown-menu-item-2' : '1000 * Heaviside(x-0.4) * Heaviside(0.6-x) ',
    'dropdown-menu-item-3' : '1e4 * (Heaviside(x-0.4) * Heaviside(0.45-x) + Heaviside(x-0.55) * Heaviside(0.6-x))',
    'dropdown-menu-item-4' : '-1e5*summation(Heaviside(1/100 - abs(x-n/5)), (n,1,4))',
    'dropdown-menu-item-5' : '-1e4 * exp(-(x-0.5)**2 / (2*0.05**2))',
    'dropdown-menu-item-6' : '1000 * sin(20*x) * x**4',
    'dropdown-menu-item-clear' : '0',
}

input_group = dbc.InputGroup(
    [
        dbc.DropdownMenu(dropdown_menu_items, label="Examples"),
        dbc.Input(
            id="func-input",
            type="text",
            value="0",
            placeholder="Sympy expression...",
        ),
    ]
)

N_MAX = 40

controls = dbc.Card(
    [
        dash.html.Div(
            [
                # dbc.Label("Potential V(x)"),
                dash.dcc.Markdown(
                    id="info-text", mathjax=True, style={"text-align": "center"}
                ),
                input_group,
            ],
            style={"margin-bottom": "20px"},
        ),
        dbc.Row(
            [
                # n_max input field
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("n_max"),
                        dbc.Input(
                            id="n-input",
                            type="number",
                            value=10,
                            min=0,
                            max=N_MAX,
                            step=1,
                            debounce=False,
                            placeholder="n_max",
                        ),
                    ],
                    style={"width": "10em"},
                ),
                # Offcanvas Info Button
                dbc.Col(
                    dash.html.Div(
                        dbc.Button(
                            "Info",
                            id="open-offcanvas",
                            n_clicks=0,
                            style={"width": "100%"},
                        ),
                        style={"text-align": "center"},
                    )
                ),
                dbc.Offcanvas(
                    dash.dcc.Markdown(
                        r"""
                ### Schrödingergleichung

                $$i\hbar\frac{\partial \psi(x,t)}{\partial t} = \hat{H} \psi(x,t)$$

                Für zeitunabhängige Hamiltonoperatoren ist die Zeitentwicklung eines Zustands

                $$\psi(x,t) = \sum_{n=0}^{\infty} c_n e^{-iE_n(t-t_0)/\hbar } \,\phi^{(n)}(x), \quad c_n = \langle \phi^{(n)}|\psi(t_0)\rangle$$

                mit orthogonalen Eigenzuständen $\phi^{(n)}$ und reellen Energieeigenwerten $E_n$ als Lösung der stationären Schrödingergleichung

                $$\hat{H}\phi = E\phi$$

                ### Eigenzustände - numerisch

                #### Räumlich beschränktes Teilchen in 1D

                $$\hat{H} = -\frac{\hbar^2}{2m}\frac{\mathrm{d}^2}{\mathrm{d}x^2} + V(x)$$

                $$\phi(0) = \phi(L) = 0$$

                $$\langle\phi|\psi\rangle = \int_0^L \phi^*(x) \psi(x) \mathrm{d}x$$

                #### Dimensionslose Größen

                - $x' = x / L, t' = \frac{t}{mL^2}, V' = mL^2V, E' = mL^2E$ (wobei im Folgenden $'$ weggelassen) und Konvention $\hbar=1$:

                $$\left[-\frac{1}{2}\frac{\mathrm{d}^2}{\mathrm{d}{x}^2} + V(x)\right]\mathclose{}\phi(x) = E\phi(x)$$

                $$\phi(0) = \phi(1) = 0$$

                #### Diskretisierung

                - $\phi_j \approx \frac{\phi_{j-1}-2\phi_j+\phi_{j+1}}{\Delta x^2}$ mit $\phi_j = \phi(j\Delta x);\: \phi_0 = \phi_N = 0;\: \Delta x = 1/N$

                - ergibt lineares System mit tridiagonaler Matrix:

                $$
                \begin{bmatrix}\frac{1}{\Delta x^2}+V_1 & -\frac{1}{2 \Delta x^2} & \hphantom{\frac{1}{\Delta x^2}+V_1} &\\[10pt]
                -\frac{1}{2 \Delta x^2} & \ddots & \ddots &\\[10pt]
                & \ddots & \ddots & -\frac{1}{2 \Delta x^2}\\[10pt]
                & \hphantom{\frac{1}{\Delta x^2}+V_1} & -\frac{1}{2 \Delta x^2} & \frac{1}{\Delta x^2}+V_{N-1}\end{bmatrix}
                \begin{bmatrix} \phi_1 \\ \vdots \\ \phi_{N-1} \end{bmatrix} = E \begin{bmatrix} \phi_1 \\ \vdots \\ \phi_{N-1} \end{bmatrix}
                $$
                """,
                        mathjax=True,
                    ),
                    id="offcanvas",
                    title="INFO",
                    is_open=False,
                    style={"width": "max-content"},
                ),
                # Download Button
                dbc.Col(
                    dash.html.Div(
                        dbc.Button(
                            "Download", id="save-button", style={"width": "100%"}
                        ),
                        style={"text-align": "center"},
                    )
                ),
                dash.dcc.Download(id="download-plot"),
            ],
            className="gap-2",
        ),
    ],
    body=True,
)

app.layout = dbc.Container(
    [
        dash.html.H1("ExPhy III - Stationäre Schrödingergleichung"),
        dash.html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(
                    dbc.Spinner(dash.dcc.Graph(id="plotly-figure", mathjax=True)), md=8
                ),
            ],
            align="center",
        ),
    ],
    fluid=True,
    style={"padding": "30px"},
)


# Choose between examples
@app.callback(
    dash.Output("func-input", "value"),
    dash.Input("dropdown-menu-item-1", "n_clicks"),
    dash.Input("dropdown-menu-item-2", "n_clicks"),
    dash.Input("dropdown-menu-item-3", "n_clicks"),
    dash.Input("dropdown-menu-item-4", "n_clicks"),
    dash.Input("dropdown-menu-item-5", "n_clicks"),
    dash.Input("dropdown-menu-item-6", "n_clicks"),
    dash.Input("dropdown-menu-item-clear", "n_clicks"),
    prevent_initial_call=True,
)
def on_button_click(n1, n2, n3, n4, n5, n6, n_clear):
    ctx = dash.callback_context

    if not ctx.triggered:
        return "0"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        return func_examples[button_id]


valid_func = True


# Display function input and check valid for func-input
@app.callback(
    dash.Output("info-text", "children"),
    dash.Output("func-input", "valid"),
    dash.Output("func-input", "invalid"),
    dash.Input("func-input", "value"),
)
def update_func_text(func_str):
    placeholder = r"Potential: Enter a Sympy expression with $x$ as the variable, or choose an example."
    try:
        x_sym = symbols("x")
        expr = parse_expr(func_str)
        expr_vars = expr.free_symbols
        if (
            len(expr_vars) > 1
            or len(expr_vars) == 1
            and next(iter(expr_vars)) != Symbol("x")
        ):
            valid_func = False
            return placeholder, valid_func, not valid_func
        else:
            valid_func = True
            return f"Potential: $V(x)={latex(expr)}$", valid_func, not valid_func
    except:
        valid_func = False
        return placeholder, valid_func, not valid_func


fig = None  # placeholder for global var


# Update the plotly figure and check valid for n-input
@app.callback(
    dash.Output("plotly-figure", "figure"),
    # dash.Output("n-input", "valid"),
    # dash.Output("n-input", "invalid"),
    dash.Input("func-input", "value"),
    dash.Input("n-input", "value"),
)
def update_plot(func_str, n_max):
    global fig, N_MAX
    valid_n = n_max <= N_MAX
    if (not valid_func) or (not valid_n):
        return dash.no_update

    x_sym = symbols("x")
    expr = parse_expr(func_str)
    latex_expr = latex(expr)
    func = lambdify(x_sym, expr)

    x, E_n, phi_n = eigenstates(func)
    show_trail = n_max <= 20
    fig = eigenstates_animation(
        x,
        func,
        E_n,
        phi_n,
        use_box_border=True,
        n_max=n_max,
        show_trail=show_trail,
        potential_latex_expr=latex_expr,
    )

    return fig  # , valid_n, not valid_n


# Save the plot
@app.callback(
    dash.Output("download-plot", "data"),
    #dash.Output("info-text", "children", allow_duplicate=True),
    dash.Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_fig(n_clicks):
    global fig
    fig.write_html(
        "plot.html", include_mathjax="cdn"
    )  # TODO: customize the filename as needed + random value
    return dash.dcc.send_file("plot.html")#, "Downloaded plot.html"


# Show Offcanvas with info
@app.callback(
    dash.Output("offcanvas", "is_open"),
    dash.Input("open-offcanvas", "n_clicks"),
    [dash.State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

app.run(debug=False)

# TODO:
# - input check does not check if functions are available! - any text with () at end is recognized as function but not invalidated - include try lambdify i.e. check if fails
# - validate n input!
# - shift phi height on energy axis
