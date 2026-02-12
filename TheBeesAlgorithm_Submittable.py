import numpy as np
import matplotlib.pyplot as plt
import time
import random as rnd
# rnd.seed(43)
# np.random.seed(43)

lowerBound = -15
upperBound = 15
bounds = [[lowerBound, upperBound], [lowerBound, upperBound]]

def CrossInTray(x):
    '''
    Definition of Cross-in-Tray function
    '''
    x0, x1 = x
    fx = -0.001 *( np.abs(np.sin(x0) * np.sin(x1) * np.exp(np.abs(100 - np.sqrt(x0**2 + x1**2) / np.pi)))+1)**0.1
    return fx

def Generate_nghk(middle, rows, cols): 
    '''
    Generates adaptive nerighbourhood scaling factor for kth patch, used in the local search
    Picks a random scale (M), builds a small interval arouond midddle point (middle). Sample one nghk value in interval 
    
    nghk = neighbourhood scaling parameter for the k-th patch (0 < nghk < 1)
    '''
    min_scale = 0.0 
    max_scale = 1.0 
    def d_triangular(min, middle, max) -> float:
        m = np.random.randint(1, 11)  # 1..10 inclusive 
        a = (middle - min) / 10.0
        c = (max - middle) / 10.0
        low = middle - m * a
        high = middle + m * c
        return float(np.random.uniform(low, high))

    M = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            M[i, j] = d_triangular(min_scale, middle, max_scale)
    return M

def exploitation(x, PatchSize, nghk, upperBound=upperBound, lowerBound=lowerBound):
    '''
    Local Search around position x, within a neighbourhood defined by PatchSize scaled by nghk
    '''
    # Scale neighbourhood
    r = nghk* PatchSize 
    n_var = x.size
    #select 1 dimension
    k=np.random.randint(0,n_var) 
    y = np.copy(x)
    # Change only selected dimension
    if np.ndim(PatchSize) > 0: 
        y[k] += np.random.uniform(-r[k], r[k])
    else:
        y[k] += np.random.uniform(-r, r)
    # Enforce bounds
    y = np.minimum(y, upperBound)
    y = np.maximum(y, lowerBound)
    return y

def incremental_kmeans_1d(x, max_K=5, n_init=5, n_iter=50):
    '''
    K=1..max_K, compute Sum_Distortion (sum of squared distances to assigned centroid), and choose K with minimum distortion.
    '''
    #clean 1d float array
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if max_K is None:
        max_K = n - 1
    # Safety: K can't be less than 1, and can't be >= n in this setup.
    max_K = max(1, min(max_K, n - 1))
    all_results = []

    #Loop over different k values (number of clusters) 
    for K in range(1, max_K + 1):
        # store the best result (lowest distortion)
        best_run_for_this_K = None  
        # For a fixed K, do multiple random restarts
        for restart in range(n_init):
            # Initialise centroids,  Pick K distinct points from x as starting centroids
            centroids = np.random.choice(x, size=K, replace=False) 

            for it in range(n_iter): 
                #assign to nearest centroid (1D)
                dists = np.abs(x[:, None] - centroids[None, :])  
                labels = np.argmin(dists, axis=1)
                new_centroids = centroids.copy()
                for k in range(K):
                    points_in_cluster = x[labels == k]
                    # If a cluster is empty, we leave its centroid unchanged.
                    if points_in_cluster.size > 0: 
                        new_centroids[k] = points_in_cluster.mean()
                # Stop early if centroids are no longer changing
                if np.allclose(new_centroids, centroids): 
                    break
                centroids = new_centroids
            # Distortion = sum over clusters of sum((point - centroid)^2)
            distortion = 0.0 
            for k in range(K):
                points_in_cluster = x[labels == k]
                if points_in_cluster.size > 0:
                    distortion += float(np.sum((points_in_cluster - centroids[k]) ** 2))

            # Keep the best restart for this K (lowest distortion)
            if best_run_for_this_K is None or distortion < best_run_for_this_K["sum_distortion"]:
                best_run_for_this_K = {
                    "K": K,
                    "centroids": centroids.copy(),
                    "labels": labels.copy(),
                    "sum_distortion": distortion,}
        all_results.append(best_run_for_this_K)
    chosen = min(all_results, key=lambda r: r["sum_distortion"])

    return {
        "all_results": all_results,                      # info for every K tested
        "chosen_K": chosen["K"],                         # best K found
        "labels_for_chosen_K": chosen["labels"],         # cluster label per point (0..K-1)
        "centroids_for_chosen_K": chosen["centroids"],   # centroid per cluster
        "sum_distortion_for_chosen_K": chosen["sum_distortion"],  # distortion value
    }

def archive_update(archive, xj, fj, bounds, max_archive=5, D_min=0.12, D_max=0.25, best_so_far_value=None):
    '''
    Function to update the archive of solutions based on distance and quality criteria.
    '''
    xj = np.asarray(xj, dtype=float)
    fj = float(fj)
    # Scaling
    bnds = np.asarray(bounds, dtype=float)
    lo = bnds[:, 0]
    hi = bnds[:, 1]
    span = hi - lo
    span = np.where(span == 0.0, 1.0, span)  # avoid divide by zero
    xj_s = (xj - lo) / span

    def dist_scaled(xi):
        xi = np.asarray(xi, dtype=float)
        xi_s = (xi - lo) / span
        d = xj_s - xi_s
        return float(np.sqrt(np.dot(d, d)))

    # Add solution if archive is empty
    if len(archive) == 0:
        archive.append({"x": xj.copy(), "f": fj})
        return archive, "added_first", {"a": None, "b": None, "D_aj": None}
    # Find nearest archived solution x_a 
    dists = [dist_scaled(item["x"]) for item in archive]
    a = int(np.argmin(dists))
    D_aj = float(dists[a])
    # Find worst archived solution x_b (largest f)
    fvals = [item["f"] for item in archive]
    b = int(np.argmax(fvals))

    # New region candidate
    if D_aj > D_min:
        # Archive not full, add solution
        if len(archive) < max_archive:
            archive.append({"x": xj.copy(), "f": fj})
            return archive, "added_first", {"a": None, "b": None, "D_aj": None}
        # Archive full, replace worst if better than worst
        if fj < archive[b]["f"]:
            archive[b] = {"x": xj.copy(), "f": fj}
            return archive, "added_new_region", {"a": a, "b": b, "D_aj": D_aj}
        # Otherwise discard
        return archive, "discarded_dissimilar_but_not_better_than_worst", {"a": a, "b": b, "D_aj": D_aj}
    
    # Not sufficiently dissimilar 
    if best_so_far_value is not None and fj <= float(best_so_far_value):
        archive[a] = {"x": xj.copy(), "f": fj}
        return archive, "replaced_nearest_best_so_far", {"a": a, "b": b, "D_aj": D_aj}
    # Rule: if close enough AND better than nearest, replace nearest
    if (D_aj <= D_max) and (fj < archive[a]["f"]):
        archive[a] = {"x": xj.copy(), "f": fj}
        return archive, "replaced_nearest_better_in_region", {"a": a, "b": b, "D_aj": D_aj}
    # Else discard
    return archive, "discarded_too_similar", {"a": a, "b": b, "D_aj": D_aj}

def BA1_implementation(population_size, max_evals): 
    '''
    Implementation of the single parameter Bees Algorithm (BA1).
    '''
    archive = []
    dim = 2
    bees = []
    counter = 0
    search_points= []
    patches = []
    opt_cost = []
    patch_size_range = np.array([upperBound - lowerBound] * dim, dtype=float)
    #Randomly initialise scout bees (uniformly in search space)
    for i in range(population_size):
        position = np.random.uniform(lowerBound, upperBound, size=dim)
        cost = CrossInTray(position)
        search_points.append(position.copy())
        counter += population_size

        bee = {
            "name": i,
            "position": position,
            "cost": float(cost),
            "size": np.copy(patch_size_range),  # patch size vector
            "stagnated": 0,
            "cluster": 0,
            "counter": counter,
            "distance": None,
        }
        bees.append(bee)

    #Sort bees by cost (best bee first)
    bees.sort(key=lambda b: b["cost"])
    best_bee = bees[0]
    #Calculate euclidean dist to best bee
    best_pos = best_bee["position"]
    for b in bees:
        b["distance"] = float(np.linalg.norm(best_pos - b["position"]))
    #Update archive
    archive, _, _ = archive_update(archive, best_bee["position"], best_bee["cost"], bounds, max_archive=5, D_min=0.12, D_max=0.25, best_so_far_value=best_bee["cost"])
    #Incremental 1D kmeans clustering
    distances = np.array([b["distance"] for b in bees], dtype=float)
    clustering = incremental_kmeans_1d(distances)
    chosen_K = clustering["chosen_K"]
    labels = clustering["labels_for_chosen_K"]

    for i, b in enumerate(bees):
        b["cluster"] = int(labels[i] + 1)
    cluster_sizes = []
    for c in range(1, chosen_K + 1):
        cluster_sizes.append(np.sum([1 for b in bees if b["cluster"] == c]))
    bee_recruitment = cluster_sizes
    
    for c in range(1, chosen_K + 1):
        # Gather bees in this cluster
        cluster_bees = [b for b in bees if b["cluster"] == c]

        # Defensive: if empty, skip
        if not cluster_bees:
            continue

        # The first in this cluster is best (bees are sorted by cost)
        cluster_bees.sort(key=lambda b: b["cost"])
        best_in_cluster = cluster_bees[0]

        patches.append({
            "position": best_in_cluster["position"].copy(),
            "cost": float(best_in_cluster["cost"]),
            "size": best_in_cluster["size"].copy(),
            "stagnated": int(best_in_cluster["stagnated"]),
            "counter": int(best_in_cluster["counter"]),
            "recruited": int(bee_recruitment[c - 1]),  # number of workers to sample around this patch
        })
    best_sol = {"cost": float("inf"), "position": None}
    ssize = np.linspace(0.0, 1.0, num=len(patches)) if patches else np.array([])

    for it in range(1, max_evals + 1):
        if counter >= max_evals:
            break
        # For each patch, recruit workers and perform foraging
        for i, patch in enumerate(patches):
            if counter >= max_evals:
                break
            recruited = patch["recruited"]
            if recruited <= 0:
                continue
            assignment = Generate_nghk(ssize[i], 1, recruited).ravel()
            best_worker = {"cost": float("inf")}

            for j in range(recruited):
                if counter >= max_evals:
                    break

                nghk = float(assignment[j])
                worker_pos = exploitation(patch["position"], patch["size"], nghk, upperBound, lowerBound)
                worker_cost = float(CrossInTray(worker_pos))
                search_points.append(worker_pos.copy())
                counter += recruited
                if worker_cost < best_worker["cost"]:
                    best_worker = {
                        "position": worker_pos,
                        "cost": worker_cost,
                        "size": patch["size"].copy(),
                        "stagnated": patch["stagnated"],
                        "recruited": recruited,
                        "counter": counter
                    }

            # Patch update rules (improve, otherwise stagnate/shrink/restart)
            if best_worker["cost"] < patch["cost"]:
                # Improvement: take the best worker as the new patch centre
                patch["position"] = best_worker["position"].copy()
                patch["cost"] = float(best_worker["cost"])
                patch["stagnated"] = 0
                patch["counter"] = int(best_worker["counter"])
            else:
                # No improvement
                patch["stagnated"] += 1

                # Shrink patch size over time - from literature
                shrink_factor = (1.0 - (3.0 * it / (4.0 * max_evals)))
                patch["size"] = patch["size"] * shrink_factor

                # If stagnated too long, re-scout this patch (random restart)
                if patch["stagnated"] >= int(round(population_size / max(1, chosen_K))):
                    patch["position"] = np.random.uniform(lowerBound, upperBound, size=dim)
                    patch["cost"] = float(CrossInTray(patch["position"]))
                    patch["size"] = np.copy(patch_size_range)
                    patch["stagnated"] = 0
                    patch["counter"] = counter

            archive, _, _ = archive_update(archive, patch["position"], patch["cost"], bounds, max_archive=5, D_min=0.12, D_max=0.25, best_so_far_value=best_sol["cost"] if best_sol["position"] is not None else patch["cost"])

        # Sort patches by cost (best first)
        patches.sort(key=lambda p: p["cost"])

        # Update best-ever solution
        if patches and patches[0]["cost"] < best_sol["cost"]:
            best_sol["cost"] = float(patches[0]["cost"])
            best_sol["position"] = patches[0]["position"].copy()

        if best_sol["position"] is not None: #update archive procided valid position
            archive, _, _ = archive_update(archive, best_sol["position"], best_sol["cost"], bounds, max_archive=5, D_min=0.12, D_max=0.25, best_so_far_value=best_sol["cost"])

        # Record best-so-far curve
        opt_cost.append(best_sol["cost"])
        iters_done = it

    return iters_done, opt_cost, counter,  best_sol["position"], np.array(search_points), archive

def plot_CrossInTray(search_points, best_pos, archive, global_minima):
    pts = np.asarray(search_points, dtype=float)
    global_minima =np.asarray(global_minima, dtype=float)

    plt.figure()
    plt.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.35)  # lots of points -> small markers


    bp = np.asarray(best_pos, dtype=float).ravel()
    plt.scatter(bp[0], bp[1], s=200, marker="X", label="Best solution", zorder=4)  # highlight best point

    A = np.array([item["x"] for item in archive], dtype=float)

    plt.scatter(A[:, 0], A[:, 1], s=80, marker="o", edgecolors="black", label="Archived optima", zorder=3)
    plt.scatter(global_minima[:, 0], global_minima[:, 1], s=50, marker="*", color="red", label="Known global minima", zorder=6)
    plt.xlim(lowerBound, upperBound)
    plt.ylim(lowerBound, upperBound)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("x0", fontsize=14)
    plt.ylabel("x1", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.show()

def plot_convergence(cost_history, ylabel="Best objective value f(x)", max_evals=1000):
    """
    Plot best-so-far objective value against iteration.
    """
    y = np.asarray(cost_history, dtype=float)
    x = np.linspace(0, max_evals, len(y))
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel("Objective function evaluations")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def count_optima_regions_in_archive(archive, global_minima, tol=0.75):
    """
    Count how many known global minima regions are represented in the archive.
    A region is counted if ANY archived solution is within 'tol' of that minimum.
    """
    if not archive:
        return 0
    A = np.array([item["x"] for item in archive], dtype=float)
    G = np.array(global_minima, dtype=float)
    found = 0
    for gm in G:
        if np.any(np.linalg.norm(A - gm, axis=1) <= tol):
            found += 1
    return found

if __name__ == "__main__":
    max_evals = 1000
    independentRuns = 500
    populationSize = 8

    # Known global minima (approximately) - for comparison
    global_minima = [
        [1.34941, 1.34941],
        [1.34941, -1.34941],
        [-1.34941, 1.34941],
        [-1.34941, -1.34941]]

    results = []
    archive = []
    best_objective_values = []
    min_distances = []
    run_times = []
    regions_found_per_run = []

    for run in range(independentRuns):
        t0 = time.perf_counter()

        iters_done, opt_cost, counter, best_position, search_points, archive = BA1_implementation(population_size=populationSize, max_evals=max_evals)

        t1 = time.perf_counter()
        run_times.append(t1 - t0)
        best_cost = opt_cost[-1] if opt_cost else None
        best_objective_values.append(best_cost)
        results.append({"run": run + 1, "iters": iters_done, "best_cost": best_cost, "nfe": counter})
        regions_found = count_optima_regions_in_archive(archive, global_minima, tol=0.75)
        regions_found_per_run.append(regions_found)
        # Distance to nearest global minimum (per run)
        best = best_position
        min_distance = float('inf')
        for gm in global_minima:
            dist = np.sqrt((best[0] - gm[0])**2 + (best[1] - gm[1])**2)
            if dist < min_distance:
                min_distance = dist
        min_distances.append(min_distance)

    # Plot final run results
    # plot_CrossInTray(search_points, best_position, archive, global_minima)
    # plot_convergence(opt_cost, ylabel="Best objective value f(x)", max_evals=max_evals)

    # Display distances to approx. global minima
    avg_min_distance = float(np.mean(min_distances)) if min_distances else None
    sd_min_distance = float(np.std(min_distances)) if min_distances else None
    print(f"\nAverages over {independentRuns} runs:")
    print(f"Average distance to nearest global minimum: {avg_min_distance:.4f}")
    print(f"Standard deviation of distance to nearest global minimum: {sd_min_distance:.4f}")
    # Display best objective function values   
    avg_best_obj = float(np.mean(best_objective_values)) if best_objective_values else float("nan")
    sd_best_obj = float(np.std(best_objective_values)) if len(best_objective_values) > 1 else 0.0
    print(f"Average best objective value: {avg_best_obj:.4f}")
    print(f"Standard deviation of best objective value: {sd_best_obj:.4f}")
    # Display regions found
    avg_regions = float(np.mean(regions_found_per_run)) if regions_found_per_run else None
    sd_regions  = float(np.std(regions_found_per_run))  if regions_found_per_run else None
    print(f"Average number of global minima regions located: {avg_regions:.3f} / 4")
    print(f"Standard deviation of global minima regions located:: {sd_regions:.3f}")
    #Display run times
    avg_time = float(np.mean(run_times)) if run_times else None
    sd_time  = float(np.std(run_times))  if run_times else None
    print(f"Average run time: {avg_time:.4f} seconds")
    print(f"Standard deviation of run time: {sd_time:.4f} seconds")