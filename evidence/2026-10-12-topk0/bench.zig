const std = @import("std");
const lanes = @import("lanes");
const sample = @import("sample");
pub fn main() !void {
    const gpa = std.heap.c_allocator;
    var rng = std.Random.DefaultPrng.init(3);
    const r = rng.random();
    const words = try gpa.alloc(u16, 248320);
    for (words) |*w| {
        const f: f32 = @floatCast(r.floatNorm(f64) * 2.5 - 2.0);
        w.* = @intCast(@as(u32, @bitCast(f)) >> 16);
    }
    for (0..20) |j| words[r.uintLessThan(usize, words.len)] = @intCast(@as(u32, @bitCast(@as(f32, 18.0 - @as(f32, @floatFromInt(j)) * 0.3))) >> 16);
    const cases = [_]lanes.Sampling{
        .{ .seed = 1, .temperature = 1.0, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 1, .temperature = 0.7, .top_k = 0, .top_p = 0.95 },
        .{ .seed = 1, .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0.05 },
        .{ .seed = 1, .temperature = 0.7, .top_k = 20, .top_p = 0.95 },
    };
    for (cases) |p| {
        var t0 = @as(i128, std.c.mach_absolute_time());
        var a: u32 = 0;
        for (0..20) |i| a = try sample.reference(gpa, words, p, i);
        const ref_ms = @as(f64, @floatFromInt(@as(i128, std.c.mach_absolute_time()) - t0)) / 20.0 / 24e3;
        t0 = @as(i128, std.c.mach_absolute_time());
        var b: u32 = 0;
        for (0..200) |i| b = try sample.draw(gpa, words, p, i);
        const new_ms = @as(f64, @floatFromInt(@as(i128, std.c.mach_absolute_time()) - t0)) / 200.0 / 24e3;
        var same = true;
        for (0..50) |i| same = same and (try sample.reference(gpa, words, p, i)) == (try sample.draw(gpa, words, p, i));
        std.debug.print("top_k {d} top_p {d} min_p {d} t {d}: reference {d:.2} ms, draw {d:.3} ms a row; equal on 50 positions: {}\n", .{ p.top_k, p.top_p, p.min_p, p.temperature, ref_ms, new_ms, same });
        std.mem.doNotOptimizeAway(a);
        std.mem.doNotOptimizeAway(b);
    }
}
