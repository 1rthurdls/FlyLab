"""Numba kernel: the same LIF update as brain.Brain.run (torch), fused per neuron.

Each trial is independent, so trials run in parallel (one per core) and the
whole time loop for a trial stays in cache-friendly compiled code.
Update order per 0.1 ms step is identical to fly-brain/code/run_pytorch.py:
  rec   = sum of w over synapses of neurons that spiked last step (not silenced)
  refrac= 0 if spiked last step else refrac+1
  g_new = g*(1-dt/tau) + delayed_rec * (refrac >= refrac_steps)
  v     = v + poisson_kick ;  v = v + dt/tau_m * (g_old - (v - v_rest))
  spike = v > v_th ;  v -> v_reset, g_new -> 0 on spike
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=False)
def run_trials(
    indptr, post, w,  # CSR by presynaptic neuron; w in mV
    N, n_steps, L,  # neurons, steps, delay-line length
    refrac_steps,  # int32 (N,)
    silenced,  # bool (N,)
    stim_idx,  # int64 (S,)
    stim_spikes,  # bool (B, n_steps, S)  precomputed Poisson input
    kick, syn_decay, mem, v0, v_rest, v_th, v_reset,
    max_events,
):
    B = stim_spikes.shape[0]
    S = stim_idx.shape[0]
    counts = np.zeros((B, N), np.int32)
    ev_step = np.zeros((B, max_events), np.int32)
    ev_neu = np.zeros((B, max_events), np.int32)
    n_ev = np.zeros(B, np.int64)
    for b in prange(B):
        v = np.full(N, v0, np.float32)
        g = np.zeros(N, np.float32)
        buf = np.zeros((L, N), np.float32)
        rec = np.zeros(N, np.float32)
        refrac = refrac_steps.copy()
        spk = np.zeros(N, np.bool_)
        stim = np.zeros(N, np.float32)
        prev = np.zeros(N, np.int32)  # neurons that spiked last step
        n_prev = 0
        cur = np.zeros(N, np.int32)
        ne = 0
        for t in range(n_steps):
            # recurrent input from last step's spikes
            for k in range(n_prev):
                pre = prev[k]
                if silenced[pre]:
                    continue
                for e in range(indptr[pre], indptr[pre + 1]):
                    rec[post[e]] += w[e]
            for k in range(S):
                if stim_spikes[b, t, k]:
                    stim[stim_idx[k]] += kick
            slot = t % L
            n_cur = 0
            for i in range(N):
                if spk[i]:
                    refrac[i] = 0
                else:
                    refrac[i] += 1
                gi = g[i]
                gn = gi * syn_decay
                if refrac[i] >= refrac_steps[i]:
                    gn += buf[slot, i]
                buf[slot, i] = rec[i]
                rec[i] = 0.0
                vi = v[i] + stim[i]
                stim[i] = 0.0
                vi = vi + mem * (gi - (vi - v_rest))
                if vi > v_th:
                    spk[i] = True
                    vi = v_reset
                    gn = 0.0
                    cur[n_cur] = i
                    n_cur += 1
                else:
                    spk[i] = False
                v[i] = vi
                g[i] = gn
            for k in range(n_cur):
                i = cur[k]
                counts[b, i] += 1
                prev[k] = i
                if ne < max_events:
                    ev_step[b, ne] = t
                    ev_neu[b, ne] = i
                    ne += 1
            n_prev = n_cur
        n_ev[b] = ne
    return counts, ev_step, ev_neu, n_ev


@njit(cache=True, nogil=True, fastmath=False)
def run_one_trial_sparse(
    indptr, post, w, N, n_steps, L, refrac_steps, silenced, stim_idx, stim_spikes_b,
    kick, syn_decay, mem, v0, v_rest, v_th, v_reset, max_events, eps,
):
    """Active-set version of run_trials for ONE trial (released GIL -> run trials in threads).

    Identical update rule, but neurons whose state is at rest (v = v_rest, g = 0,
    no pending delayed input, out of refractory) are skipped, because the update
    would leave them unchanged. A neuron returns to the resting set once
    |v - v_rest| < eps and |g| < eps (eps = 1e-4 mV by default: negligible vs the
    7 mV gap between rest and threshold; float32 resolution at -52 mV is ~4e-6 mV,
    so eps must be larger than that). Validated spike-for-spike against run_trials.
    """
    S = stim_idx.shape[0]
    counts = np.zeros(N, np.int32)
    ev_step = np.zeros(max_events, np.int32)
    ev_neu = np.zeros(max_events, np.int32)
    # Neurons at rest have no stored state. Active neurons get a compact "slot"
    # so their state is contiguous in memory (fits in L2 cache).
    slot_of = np.full(N, -1, np.int32)
    s_neu = np.zeros(N, np.int32)
    s_v = np.zeros(N, np.float32)
    s_g = np.zeros(N, np.float32)
    s_rec = np.zeros(N, np.float32)
    s_stim = np.zeros(N, np.float32)
    s_ref = np.zeros(N, np.int32)
    s_rs = np.zeros(N, np.int32)
    s_last = np.zeros(N, np.int32)
    s_spk = np.zeros(N, np.bool_)
    s_buf = np.zeros((N, L), np.float32)
    n_act = 0
    prev = np.zeros(N, np.int32)
    cur = np.zeros(N, np.int32)
    n_prev = 0
    ne = 0
    act_sum = 0
    for t in range(n_steps):
        for k in range(n_prev):
            pre = prev[k]
            if silenced[pre]:
                continue
            for e in range(indptr[pre], indptr[pre + 1]):
                j = post[e]
                sl = slot_of[j]
                if sl < 0:
                    sl = n_act
                    n_act += 1
                    slot_of[j] = sl
                    s_neu[sl] = j
                    s_v[sl] = v_rest
                    s_g[sl] = 0.0
                    s_rec[sl] = 0.0
                    s_stim[sl] = 0.0
                    s_ref[sl] = refrac_steps[j] + 1000
                    s_rs[sl] = refrac_steps[j]
                    s_spk[sl] = False
                    for q in range(L):
                        s_buf[sl, q] = 0.0
                s_rec[sl] += w[e]
                s_last[sl] = t
        for k in range(S):
            if stim_spikes_b[t, k]:
                j = stim_idx[k]
                sl = slot_of[j]
                if sl < 0:
                    sl = n_act
                    n_act += 1
                    slot_of[j] = sl
                    s_neu[sl] = j
                    s_v[sl] = v_rest
                    s_g[sl] = 0.0
                    s_rec[sl] = 0.0
                    s_stim[sl] = 0.0
                    s_ref[sl] = refrac_steps[j] + 1000
                    s_rs[sl] = refrac_steps[j]
                    s_last[sl] = -1000000
                    s_spk[sl] = False
                    for q in range(L):
                        s_buf[sl, q] = 0.0
                s_stim[sl] += kick
        slot = t % L
        n_cur = 0
        act_sum += n_act
        a = n_act - 1
        while a >= 0:  # downward, so swap-with-last removal is safe
            if s_spk[a]:
                s_ref[a] = 0
            else:
                s_ref[a] += 1
            gi = s_g[a]
            gn = gi * syn_decay
            if s_ref[a] >= s_rs[a]:
                gn += s_buf[a, slot]
            s_buf[a, slot] = s_rec[a]
            s_rec[a] = 0.0
            vi = s_v[a] + s_stim[a]
            s_stim[a] = 0.0
            vi = vi + mem * (gi - (vi - v_rest))
            if vi > v_th:
                s_spk[a] = True
                vi = v_reset
                gn = 0.0
                cur[n_cur] = s_neu[a]
                n_cur += 1
            else:
                s_spk[a] = False
            s_v[a] = vi
            s_g[a] = gn
            if (not s_spk[a]) and abs(vi - v_rest) < eps and abs(gn) < eps                     and t - s_last[a] >= L and s_ref[a] >= s_rs[a]:
                # back at rest: free the slot (move last slot here)
                slot_of[s_neu[a]] = -1
                z = n_act - 1
                if z != a:
                    s_neu[a] = s_neu[z]
                    s_v[a] = s_v[z]
                    s_g[a] = s_g[z]
                    s_rec[a] = s_rec[z]
                    s_stim[a] = s_stim[z]
                    s_ref[a] = s_ref[z]
                    s_rs[a] = s_rs[z]
                    s_last[a] = s_last[z]
                    s_spk[a] = s_spk[z]
                    for q in range(L):
                        s_buf[a, q] = s_buf[z, q]
                    slot_of[s_neu[a]] = a
                n_act -= 1
            a -= 1
        # same spike order as the dense kernel -> bit-identical sums
        srt = np.sort(cur[:n_cur])
        for k in range(n_cur):
            i = srt[k]
            prev[k] = i
            counts[i] += 1
            if ne < max_events:
                ev_step[ne] = t
                ev_neu[ne] = i
                ne += 1
        n_prev = n_cur
    return counts, ev_step[:ne].copy(), ev_neu[:ne].copy(), ne, act_sum / max(n_steps, 1)


@njit(cache=True, nogil=True, fastmath=False)
def run_chunk_sparse(
    indptr, post, w, N, t_start, n_steps, L, refrac_steps, silenced, stim_idx, stim_spikes_b,
    kick, syn_decay, mem, v0, v_rest, v_th, v_reset, max_events, eps,
    slot_of, s_neu, s_v, s_g, s_rec, s_stim, s_ref, s_rs, s_last, s_spk, s_buf, prev, cur, scal,
):
    """Same update as run_one_trial_sparse, but the network state lives in the arrays passed in and is
    updated in place, so the brain can be advanced chunk by chunk (closed loop with a body/world).
    scal = int64[2] = (n_act, n_prev). Running n chunks back to back is spike-identical to one long run
    with the same input spikes (validated in scripts/validate_stepper.py)."""
    S = stim_idx.shape[0]
    counts = np.zeros(N, np.int32)
    ev_step = np.zeros(max_events, np.int32)
    ev_neu = np.zeros(max_events, np.int32)
    n_act = scal[0]
    n_prev = scal[1]
    ne = 0
    for tt in range(n_steps):
        t = t_start + tt
        for k in range(n_prev):
            pre = prev[k]
            if silenced[pre]:
                continue
            for e in range(indptr[pre], indptr[pre + 1]):
                j = post[e]
                sl = slot_of[j]
                if sl < 0:
                    sl = n_act
                    n_act += 1
                    slot_of[j] = sl
                    s_neu[sl] = j
                    s_v[sl] = v_rest
                    s_g[sl] = 0.0
                    s_rec[sl] = 0.0
                    s_stim[sl] = 0.0
                    s_ref[sl] = refrac_steps[j] + 1000
                    s_rs[sl] = refrac_steps[j]
                    s_spk[sl] = False
                    for q in range(L):
                        s_buf[sl, q] = 0.0
                s_rec[sl] += w[e]
                s_last[sl] = t
        for k in range(S):
            if stim_spikes_b[tt, k]:
                j = stim_idx[k]
                sl = slot_of[j]
                if sl < 0:
                    sl = n_act
                    n_act += 1
                    slot_of[j] = sl
                    s_neu[sl] = j
                    s_v[sl] = v_rest
                    s_g[sl] = 0.0
                    s_rec[sl] = 0.0
                    s_stim[sl] = 0.0
                    s_ref[sl] = refrac_steps[j] + 1000
                    s_rs[sl] = refrac_steps[j]
                    s_last[sl] = -1000000
                    s_spk[sl] = False
                    for q in range(L):
                        s_buf[sl, q] = 0.0
                s_stim[sl] += kick
        slot = t % L
        n_cur = 0
        a = n_act - 1
        while a >= 0:
            if s_spk[a]:
                s_ref[a] = 0
            else:
                s_ref[a] += 1
            gi = s_g[a]
            gn = gi * syn_decay
            if s_ref[a] >= s_rs[a]:
                gn += s_buf[a, slot]
            s_buf[a, slot] = s_rec[a]
            s_rec[a] = 0.0
            vi = s_v[a] + s_stim[a]
            s_stim[a] = 0.0
            vi = vi + mem * (gi - (vi - v_rest))
            if vi > v_th:
                s_spk[a] = True
                vi = v_reset
                gn = 0.0
                cur[n_cur] = s_neu[a]
                n_cur += 1
            else:
                s_spk[a] = False
            s_v[a] = vi
            s_g[a] = gn
            if (not s_spk[a]) and abs(vi - v_rest) < eps and abs(gn) < eps \
                    and t - s_last[a] >= L and s_ref[a] >= s_rs[a]:
                slot_of[s_neu[a]] = -1
                z = n_act - 1
                if z != a:
                    s_neu[a] = s_neu[z]
                    s_v[a] = s_v[z]
                    s_g[a] = s_g[z]
                    s_rec[a] = s_rec[z]
                    s_stim[a] = s_stim[z]
                    s_ref[a] = s_ref[z]
                    s_rs[a] = s_rs[z]
                    s_last[a] = s_last[z]
                    s_spk[a] = s_spk[z]
                    for q in range(L):
                        s_buf[a, q] = s_buf[z, q]
                    slot_of[s_neu[a]] = a
                n_act -= 1
            a -= 1
        srt = np.sort(cur[:n_cur])
        for k in range(n_cur):
            i = srt[k]
            prev[k] = i
            counts[i] += 1
            if ne < max_events:
                ev_step[ne] = t
                ev_neu[ne] = i
                ne += 1
        n_prev = n_cur
    scal[0] = n_act
    scal[1] = n_prev
    return counts, ev_step[:ne].copy(), ev_neu[:ne].copy()


@njit(cache=True, parallel=True, fastmath=False)
def run_chunk_dense_par(
    indptr, post, w, N, n_steps, L, t_start, refrac_steps, silenced, stim_idx, stim_spikes_b,
    kick, syn_decay, mem, v_rest, v_th, v_reset, max_events,
    v, g, buf, rec, refrac, spk, prev, scal,
):
    """Dense update of run_trials (the reference kernel) for ONE trial, chunk by chunk, with the per-neuron
    update split across threads. The recurrent sum stays serial and in sorted spike order, so floating point
    order (and therefore every spike) is identical to run_trials. State is updated in place; scal[0] = n_prev.
    Used for closed-loop runs where tens of thousands of neurons are subthreshold-active at once."""
    S = stim_idx.shape[0]
    counts = np.zeros(N, np.int32)
    ev_step = np.zeros(max_events, np.int32)
    ev_neu = np.zeros(max_events, np.int32)
    stim = np.zeros(N, np.float32)
    BS = 4096
    NB = (N + BS - 1) // BS
    bcnt = np.zeros(NB, np.int64)
    bspk = np.zeros(N, np.int32)  # spikes of block b stored from b*BS (block-local, in index order)
    n_prev = scal[0]
    ne = 0
    for tt in range(n_steps):
        t = t_start + tt
        for k in range(n_prev):
            pre = prev[k]
            if silenced[pre]:
                continue
            for e in range(indptr[pre], indptr[pre + 1]):
                rec[post[e]] += w[e]
        for k in range(S):
            if stim_spikes_b[tt, k]:
                stim[stim_idx[k]] += kick
        slot = t % L
        for b in prange(NB):
          nb = 0
          for i in range(b * BS, min(N, (b + 1) * BS)):
            if spk[i]:
                refrac[i] = 0
            else:
                refrac[i] += 1
            gi = g[i]
            gn = gi * syn_decay
            if refrac[i] >= refrac_steps[i]:
                gn += buf[slot, i]
            buf[slot, i] = rec[i]
            rec[i] = 0.0
            vi = v[i] + stim[i]
            stim[i] = 0.0
            vi = vi + mem * (gi - (vi - v_rest))
            if vi > v_th:
                spk[i] = True
                vi = v_reset
                gn = 0.0
                bspk[b * BS + nb] = i
                nb += 1
            else:
                spk[i] = False
            v[i] = vi
            g[i] = gn
          bcnt[b] = nb
        n_cur = 0
        for b in range(NB):
            for k in range(bcnt[b]):
                i = bspk[b * BS + k]
                prev[n_cur] = i
                n_cur += 1
                counts[i] += 1
                if ne < max_events:
                    ev_step[ne] = t
                    ev_neu[ne] = i
                    ne += 1
        n_prev = n_cur
    scal[0] = n_prev
    return counts, ev_step[:ne].copy(), ev_neu[:ne].copy()
