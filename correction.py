import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# RouteWise: Transportation Problem Solver using VAM and MODI
# Dependencies: streamlit, pandas, numpy

st.set_page_config(
    page_title="RouteWise | Transportation Optimizer",
    page_icon="🚌",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 38px; font-weight: 800; color: #174A7E; text-align: center;}
    .sub-title {text-align: center; color: #65758B; font-size: 17px; margin-bottom: 24px;}
    .section-title {color: #174A7E; font-size: 23px; font-weight: 700;}
    div.stButton > button {width: 100%; border-radius: 10px; min-height: 46px; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="main-title">🚌 RouteWise</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">College Bus Transportation Optimizer | VAM + MODI</div>',
    unsafe_allow_html=True,
)
st.write(
    "Enter a transportation cost matrix, source supplies, and destination demands. "
    "The application calculates an initial solution using Vogel’s Approximation Method "
    "(VAM), then improves it with the MODI method."
)


def complete_basis(allocation, eps=1e-9):
    """Extend positive allocation cells to a spanning-tree basis."""
    m, n = allocation.shape
    basis = {(i, j) for i in range(m) for j in range(n) if allocation[i, j] > eps}
    parent = list(range(m + n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    # Positive cells should form a forest for a non-degenerate feasible allocation.
    # If a numerical/tie case produces a cycle, retain a spanning subset as basis.
    basis_forest = set()
    for i, j in sorted(basis):
        if union(i, m + j):
            basis_forest.add((i, j))

    basis = basis_forest
    for i in range(m):
        for j in range(n):
            if (i, j) not in basis and union(i, m + j):
                basis.add((i, j))

    if len(basis) != m + n - 1:
        raise RuntimeError("Could not construct a valid MODI basis.")
    return basis


def vam(cost, supply, demand):
    """Vogel's Approximation Method for a balanced transportation problem."""
    cost = np.asarray(cost, dtype=float)
    remaining_supply = np.asarray(supply, dtype=float).copy()
    remaining_demand = np.asarray(demand, dtype=float).copy()
    m, n = cost.shape
    allocation = np.zeros((m, n), dtype=float)
    active_rows = set(range(m))
    active_cols = set(range(n))
    steps = []
    eps = 1e-9

    while active_rows and active_cols:
        candidates = []

        for i in sorted(active_rows):
            values = sorted(cost[i, j] for j in active_cols)
            penalty = values[1] - values[0] if len(values) >= 2 else values[0]
            min_cost = values[0]
            candidates.append((penalty, -min_cost, "row", i))

        for j in sorted(active_cols):
            values = sorted(cost[i, j] for i in active_rows)
            penalty = values[1] - values[0] if len(values) >= 2 else values[0]
            min_cost = values[0]
            candidates.append((penalty, -min_cost, "col", j))

        # Deterministic tie rule: highest penalty, then lowest minimum cost,
        # then row before column, then smallest index.
        candidates.sort(
            key=lambda x: (x[0], x[1], x[2] == "row", -x[3]),
            reverse=True,
        )
        penalty, _, kind, index = candidates[0]

        if kind == "row":
            i = index
            j = min(active_cols, key=lambda col: (cost[i, col], col))
        else:
            j = index
            i = min(active_rows, key=lambda row: (cost[row, j], row))

        quantity = min(remaining_supply[i], remaining_demand[j])
        if quantity <= eps:
            if remaining_supply[i] <= eps:
                active_rows.discard(i)
            if remaining_demand[j] <= eps:
                active_cols.discard(j)
            continue

        allocation[i, j] += quantity
        remaining_supply[i] -= quantity
        remaining_demand[j] -= quantity

        steps.append({
            "Chosen line": f"{kind.title()} {index + 1}",
            "Penalty": float(penalty),
            "Source index": i,
            "Destination index": j,
            "Unit cost": float(cost[i, j]),
            "Allocation": float(quantity),
            "Supply remaining": float(remaining_supply[i]),
            "Demand remaining": float(remaining_demand[j]),
        })

        row_done = remaining_supply[i] <= eps
        col_done = remaining_demand[j] <= eps

        if row_done:
            remaining_supply[i] = 0
            active_rows.discard(i)
        if col_done:
            remaining_demand[j] = 0
            active_cols.discard(j)

    if np.any(remaining_supply > 1e-6) or np.any(remaining_demand > 1e-6):
        raise RuntimeError("VAM did not satisfy all supply and demand values.")

    return allocation, steps


def compute_potentials(cost, basis):
    """Compute MODI potentials u and v from u_i + v_j = c_ij."""
    m, n = cost.shape
    u = [None] * m
    v = [None] * n

    # A valid basis is a spanning tree, so propagation determines all potentials.
    u[0] = 0.0
    changed = True
    while changed:
        changed = False
        for i, j in basis:
            if u[i] is not None and v[j] is None:
                v[j] = cost[i, j] - u[i]
                changed = True
            elif v[j] is not None and u[i] is None:
                u[i] = cost[i, j] - v[j]
                changed = True

    if any(value is None for value in u) or any(value is None for value in v):
        raise RuntimeError("MODI basis is disconnected; potentials could not be computed.")

    return np.asarray(u, dtype=float), np.asarray(v, dtype=float)


def tree_path(basis, start, target, m, n):
    """Find the unique path between a row node and column node in the basis tree."""
    adjacency = {node: [] for node in range(m + n)}
    for i, j in basis:
        a, b = i, m + j
        adjacency[a].append((b, (i, j)))
        adjacency[b].append((a, (i, j)))

    stack = [(start, None, [])]
    visited = set()
    while stack:
        node, parent, path_edges = stack.pop()
        if node == target:
            return path_edges
        if node in visited:
            continue
        visited.add(node)
        for nxt, cell in adjacency[node]:
            if nxt != parent and nxt not in visited:
                stack.append((nxt, node, path_edges + [cell]))
    return None


def modi(cost, initial_allocation, max_iterations=1000, eps=1e-9):
    """Transportation simplex/MODI improvement with a tree basis."""
    cost = np.asarray(cost, dtype=float)
    allocation = np.asarray(initial_allocation, dtype=float).copy()
    m, n = cost.shape
    basis = complete_basis(allocation, eps)
    history = []

    for iteration in range(1, max_iterations + 1):
        u, v = compute_potentials(cost, basis)
        reduced = cost - u[:, None] - v[None, :]

        nonbasic = [
            (i, j)
            for i in range(m)
            for j in range(n)
            if (i, j) not in basis
        ]
        if not nonbasic:
            return allocation, history, True

        # Bland-style deterministic entering choice among negative reduced costs:
        # most negative first, then row/column order.
        entering = min(nonbasic, key=lambda cell: (reduced[cell], cell[0], cell[1]))
        if reduced[entering] >= -eps:
            return allocation, history, True

        i_enter, j_enter = entering
        # Adding entering edge creates one cycle. The tree path alternates row/column.
        path_edges = tree_path(basis, i_enter, m + j_enter, m, n)
        if path_edges is None or len(path_edges) % 2 != 1:
            return allocation, history, False

        # Cycle: entering cell is '+'; path edges alternate '-', '+', '-',...
        cycle = [entering] + list(reversed(path_edges))
        plus_cells = cycle[0::2]
        minus_cells = cycle[1::2]

        theta = min(allocation[cell] for cell in minus_cells)
        leaving_candidates = [
            cell for cell in minus_cells if allocation[cell] <= theta + eps
        ]
        leaving = min(leaving_candidates)

        history.append({
            "Iteration": iteration,
            "Entering cell": entering,
            "Opportunity cost": float(reduced[entering]),
            "Adjustment amount": float(theta),
            "Cycle": cycle.copy(),
        })

        for cell in plus_cells:
            allocation[cell] += theta
        for cell in minus_cells:
            allocation[cell] -= theta

        allocation[np.abs(allocation) < eps] = 0.0
        basis.remove(leaving)
        basis.add(entering)

        if len(basis) != m + n - 1:
            return allocation, history, False

    return allocation, history, False


# -------------------- USER INPUT FORM --------------------
# This project uses the user's fixed 5 x 5 college-bus transportation matrix.
m, n = 5, 5
DEFAULT_SOURCES = ["Bodi", "Chinnamanur", "Aundipatti", "Usilampatti", "Cumbum/Gudalur"]
DEFAULT_DESTINATIONS = ["C1", "C2", "C3", "C4", "C5"]
DEFAULT_COSTS = [
    [15, 10, 12, 8, 5],
    [12, 15, 10, 8, 6],
    [10, 8, 15, 12, 5],
    [8, 12, 10, 15, 6],
    [5, 8, 12, 10, 15],
]
DEFAULT_SUPPLY = [50, 50, 50, 50, 50]
DEFAULT_DEMAND = [50, 45, 55, 40, 60]

st.markdown('<div class="section-title">1. Configure your 5 × 5 college-bus problem</div>', unsafe_allow_html=True)
st.info("The project is fixed to 5 sources and 5 destinations. You can edit the names, costs, supply, and demand below.")

with st.form("transportation_input_form"):
    st.subheader("Source names")
    source_names = []
    cols = st.columns(5)
    for i in range(m):
        with cols[i]:
            source_names.append(
                st.text_input(
                    f"Source {i + 1}",
                    value=DEFAULT_SOURCES[i],
                    key=f"src_fixed_{i}",
                ).strip()
            )

    st.subheader("Destination names")
    destination_names = []
    cols = st.columns(5)
    for j in range(n):
        with cols[j]:
            destination_names.append(
                st.text_input(
                    f"Destination {j + 1}",
                    value=DEFAULT_DESTINATIONS[j],
                    key=f"dst_fixed_{j}",
                ).strip()
            )

    st.subheader("Transportation cost matrix")
    default_cost = pd.DataFrame(
        DEFAULT_COSTS,
        index=source_names,
        columns=destination_names,
    )
    cost_input = st.data_editor(
        default_cost,
        key="cost_fixed_5x5",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            col: st.column_config.NumberColumn(col, min_value=0, step=1, required=True)
            for col in default_cost.columns
        },
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Supply")
        supply_input = st.data_editor(
            pd.DataFrame({"Source": source_names, "Supply": DEFAULT_SUPPLY}),
            key="supply_fixed_5",
            num_rows="fixed",
            hide_index=True,
            disabled=["Source"],
            use_container_width=True,
            column_config={"Supply": st.column_config.NumberColumn("Supply", min_value=0, step=1, required=True)},
        )
    with right:
        st.subheader("Demand")
        demand_input = st.data_editor(
            pd.DataFrame({"Destination": destination_names, "Demand": DEFAULT_DEMAND}),
            key="demand_fixed_5",
            num_rows="fixed",
            hide_index=True,
            disabled=["Destination"],
            use_container_width=True,
            column_config={"Demand": st.column_config.NumberColumn("Demand", min_value=0, step=1, required=True)},
        )

    submitted = st.form_submit_button("🚀 Calculate using VAM and MODI")


if submitted:
    try:
        costs = cost_input.to_numpy(dtype=float)
        supplies = supply_input["Supply"].to_numpy(dtype=float)
        demands = demand_input["Demand"].to_numpy(dtype=float)

        if any(not x for x in source_names) or any(not x for x in destination_names):
            st.error("Every source and destination must have a name.")
        elif len(set(source_names)) != m or len(set(destination_names)) != n:
            st.error("Source names must be unique. Destination names must also be unique.")
        elif not (np.isfinite(costs).all() and np.isfinite(supplies).all() and np.isfinite(demands).all()):
            st.error("Please enter valid finite numbers in every cell.")
        elif (costs < 0).any() or (supplies < 0).any() or (demands < 0).any():
            st.error("Costs, supply, and demand cannot be negative.")
        elif not np.allclose(costs, np.round(costs)):
            st.error("Enter whole-number transportation costs.")
        elif not np.allclose(supplies, np.round(supplies)) or not np.allclose(demands, np.round(demands)):
            st.error("Enter whole-number supply and demand quantities.")
        elif not np.isclose(supplies.sum(), demands.sum()):
            st.error(
                f"The problem is unbalanced: total supply = {supplies.sum():g}, "
                f"total demand = {demands.sum():g}. Add a dummy source or destination "
                "with suitable costs before solving."
            )
        elif supplies.sum() <= 0:
            st.error("Total supply and demand must be greater than zero.")
        else:
            st.session_state["routewise_input"] = {
                "costs": costs.copy(),
                "supplies": supplies.copy(),
                "demands": demands.copy(),
                "sources": source_names.copy(),
                "destinations": destination_names.copy(),
            }
            st.session_state.pop("routewise_results", None)

    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"Please check your entries. Details: {exc}")


# -------------------- CALCULATE AND DISPLAY --------------------
if "routewise_input" in st.session_state:
    data = st.session_state["routewise_input"]
    costs = data["costs"]
    supplies = data["supplies"]
    demands = data["demands"]
    sources = data["sources"]
    destinations = data["destinations"]

    try:
        vam_allocation, vam_steps = vam(costs, supplies, demands)
        optimal_allocation, modi_steps, optimality_confirmed = modi(costs, vam_allocation)

        # Check feasibility and cost before displaying.
        if not np.allclose(optimal_allocation.sum(axis=1), supplies):
            raise RuntimeError("The calculated row allocations do not match supply.")
        if not np.allclose(optimal_allocation.sum(axis=0), demands):
            raise RuntimeError("The calculated column allocations do not match demand.")
        if (optimal_allocation < -1e-8).any():
            raise RuntimeError("A negative allocation was produced.")

        vam_cost = float(np.sum(vam_allocation * costs))
        optimal_cost = float(np.sum(optimal_allocation * costs))

        st.divider()
        st.markdown('<div class="section-title">2. Input transportation table</div>', unsafe_allow_html=True)
        input_table = pd.DataFrame(costs.astype(int), index=sources, columns=destinations)
        input_table["Supply"] = supplies.astype(int)
        demand_row = {destinations[j]: int(demands[j]) for j in range(len(destinations))}
        demand_row["Supply"] = ""
        input_table.loc["Demand"] = pd.Series(demand_row)
        st.dataframe(input_table, use_container_width=True)

        st.markdown('<div class="section-title">3. VAM initial solution</div>', unsafe_allow_html=True)
        st.caption("VAM selects the row or column with the largest penalty, then allocates to a lowest-cost cell.")
        vam_table = pd.DataFrame(vam_allocation.astype(int), index=sources, columns=destinations)
        st.dataframe(vam_table, use_container_width=True)
        st.success(f"VAM initial transportation cost: ₹{vam_cost:,.2f}")

        if vam_steps:
            vam_log = pd.DataFrame(vam_steps)
            vam_log["Source"] = vam_log["Source index"].map(lambda i: sources[int(i)])
            vam_log["Destination"] = vam_log["Destination index"].map(lambda j: destinations[int(j)])
            vam_log = vam_log[[
                "Chosen line", "Penalty", "Source", "Destination",
                "Unit cost", "Allocation", "Supply remaining", "Demand remaining"
            ]]
            st.subheader("VAM allocation steps")
            st.dataframe(vam_log, hide_index=True, use_container_width=True)

        st.markdown('<div class="section-title">4. MODI optimality test and improvement</div>', unsafe_allow_html=True)
        if optimality_confirmed:
            st.success("MODI terminated with all non-basic opportunity costs non-negative.")
        else:
            st.warning("The algorithm could not confirm optimality within its safeguards. Do not report this result as verified optimal.")

        final_table = pd.DataFrame(optimal_allocation.astype(int), index=sources, columns=destinations)
        st.subheader("Final allocation")
        st.dataframe(final_table, use_container_width=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("VAM initial cost", f"₹{vam_cost:,.2f}")
        k2.metric("Cost reduction", f"₹{vam_cost - optimal_cost:,.2f}")
        k3.metric("Final cost", f"₹{optimal_cost:,.2f}")

        st.subheader("MODI iteration details")
        if modi_steps:
            for step in modi_steps:
                ei, ej = step["Entering cell"]
                with st.expander(f"Iteration {step['Iteration']}"):
                    st.write(f"Entering cell: {sources[ei]} → {destinations[ej]}")
                    st.write(f"Opportunity cost (Δ): {step['Opportunity cost']:.4f}")
                    st.write(f"Adjustment quantity (θ): {step['Adjustment amount']:g}")
                    st.write("Closed loop:")
                    st.write(" → ".join(f"({sources[i]}, {destinations[j]})" for i, j in step["Cycle"]))
        else:
            st.write("No improving MODI pivot was needed.")

        # Download final allocation and a separate cost summary.
        export_table = final_table.copy()
        export_table["Supply"] = supplies.astype(int)
        demand_export = {destinations[j]: int(demands[j]) for j in range(n)}
        demand_export["Supply"] = ""
        export_table.loc["Demand"] = pd.Series(demand_export)

        st.markdown('<div class="section-title">5. Download results</div>', unsafe_allow_html=True)
        st.download_button(
            "📥 Download final allocation (CSV)",
            data=export_table.to_csv().encode("utf-8"),
            file_name="routewise_final_allocation.csv",
            mime="text/csv",
        )

        summary = pd.DataFrame({
            "Metric": ["VAM initial cost", "Cost reduction by MODI", "Final transportation cost", "Optimality confirmed"],
            "Value": [vam_cost, vam_cost - optimal_cost, optimal_cost, "Yes" if optimality_confirmed else "No"],
        })
        st.download_button(
            "📥 Download cost summary (CSV)",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="routewise_cost_summary.csv",
            mime="text/csv",
        )

    except (RuntimeError, ValueError, IndexError, KeyError) as exc:
        st.error(f"Unable to complete the calculation safely: {exc}")
        st.info("Check the input values. If the problem is valid, share this error so the algorithm can be corrected.")

st.divider()
st.caption("RouteWise | College Bus Transportation Optimization | Python • Streamlit • VAM • MODI")
