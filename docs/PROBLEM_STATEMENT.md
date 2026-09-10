# Problem Statement

## Context

**Zarrin Cake & Confectionery Industries Co.** is a mid-size industrial food
manufacturer producing cakes, cookies, wafers, and traditional pastries at
scale, running five production lines and roughly 200 employees. Its product
range spans everyday items (sponge cakes, wafers, biscuits) and
culturally-anchored seasonal specialties — most notably **kolucheh**, a
traditional Persian cookie whose demand explodes in the weeks before
**Nowruz** (Persian New Year), with a secondary spike around **Yalda
Night**.

That seasonality is the whole problem. A flat annual production plan wastes
capacity for 48 weeks and falls badly short for 2–3. The planning team
needs a repeatable process that:

1. Forecasts demand per product, weeks ahead, actually capturing the
   Nowruz/Yalda cycle rather than smoothing it away.
2. Turns that forecast into a **production plan** each of the five lines
   can physically execute — respecting machine-hour capacity *and* labor
   availability, and making a deliberate, defensible call about which
   products get priority when a line can't make everything the forecast
   calls for.
3. Turns the production plan into a **material requirements plan** —
   what raw materials need to be bought, how much, and by when, given
   each supplier's lead time — early enough that a 12-day lead-time
   ingredient doesn't become the reason a Nowruz order can't ship.
4. Flags all of this **before** it becomes a fire drill: a capacity
   shortfall or a material stockout should show up as a number in a report
   weeks in advance, not as a missed shipment.
5. Ties supply and production planning back to **working capital**: how
   long cash is tied up in raw materials, how slowly customers pay, and how
   quickly suppliers must be paid, all roll up into the **Cash Conversion
   Cycle (CCC)** — a number finance tracks independently of, but directly
   affected by, the same inventory and purchasing decisions this engine
   makes.

## Goal

Build a **Sales & Operations Planning (S&OP) engine**: a repeatable
pipeline from historical demand to a capacity- and labor-constrained
production plan and a material requirements plan, with a single
consolidated weekly report a planning meeting could actually use.

Concretely, the pipeline must:

- **Forecast weekly demand per SKU**, using enough history to model
  yearly seasonality, and report a real backtested accuracy number rather
  than an unverified claim.
- **Allocate production across 5 lines under two simultaneous constraints**
  — machine-hours (net of realistic equipment effectiveness) and
  labor-hours (from actual workforce availability, not a headcount
  fantasy) — prioritizing which products get made when the two don't add
  up to full demand.
- **Explode the plan into material requirements via a bill of materials**,
  net that against current inventory, and recommend purchase orders sized
  and timed to each material's supplier lead time and safety-stock policy.
- **Surface a single weekly summary**: forecast vs. planned units, fill
  rate, revenue captured vs. revenue at risk, and how many materials are
  running below their safety stock — the artifact an actual S&OP meeting
  would review.
- **Compute the Cash Conversion Cycle and its components (DIO, DSO, DPO)**
  as a real trailing time series — not a single asserted number — from
  simulated inventory, receivables, and payables ledgers, so the working-
  capital impact of supply and production decisions is measured, not
  assumed.
- **Run entirely on synthetic but realistic data** so the repository is
  self-contained: a data generator produces ~180K rows of internally
  consistent HR, production, sales, inventory, and purchasing history with
  genuine Nowruz/Yalda seasonality baked in, so the whole pipeline can be
  run and inspected with zero external dependencies.

## Non-goals

- Real-time replanning — this is a weekly-cycle batch planning process, not
  a live MES/scheduling system.
- Multi-line product flexibility — each SKU is produced on a single primary
  line in this model; a plant that could route a product to more than one
  line is a natural extension, not covered here.
- A UI — outputs are a warehouse table + Parquet, ready to plug into Power
  BI or any BI tool, not a bespoke dashboard app.

## Expected output

A `fact_sop_weekly_summary` table with, per week: total forecast units,
allocated (planned) units, unmet units, fill rate, revenue captured,
revenue at risk, and the count of materials below safety stock — plus the
full supporting detail: `fact_demand_forecast`, `fact_production_plan`
(per line/product/week, with the LP's allocation), `fact_material_requirements`
(per material/week, with purchase recommendations), `fact_forecast_accuracy`
(the backtest), and `fact_working_capital` (weekly DIO / DSO / DPO / CCC).
