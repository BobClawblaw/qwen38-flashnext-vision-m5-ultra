top_k 0 sampler: engine add3da3 (before) and eb219d2a (after), served alone at the recipe's settings.
Served tok/s, 300-token replies: top_k 0 t=1 23.2 -> 105.1; top_k 0 top_p 0.95 t=0.7 23.2 -> 90.7;
top_k 0 min_p 0.05 t=1 29.6 -> 118.5; top_k 500 131.9 -> 134.5; default 131.9 -> 131.8. Token shas 5/5 equal.
bench.zig, 248,320-token rows: reference 13.7-16.8 ms a row, draw 0.33-0.65 ms; top_k 20 0.28 -> 0.13 ms.
