from __future__ import division
import numpy as np
import logging
from synge.shape import verify_shape


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s ",
    level=logging.INFO,
)


def fact_apply_chev(
    clus_cfg,
    cluster,
    samples,
    label,
    data,
    indexes_overLap,
    percent,
    l1,
    l2,
    sign2,
    sign,
):
    if sign2 == "+":
        xd = xd + clus_cfg.par_shapes[l1][1] * 0.01
    else:
        xd = xd - clus_cfg.par_shapes[l1][1] * 0.01
    d_ = compute_d(clus_cfg, data, indexes_overLap, l2, xd, sign)
    result, P = common_code(
        clus_cfg, cluster, samples, label, data, d_, indexes_overLap, l1
    )
    xdmin = cdt(P, percent, result, xd)
    return xd, result, xdmin


def compute_xd(clus_cfg, l1, sign):
    if sign == "+":
        xd = clus_cfg._centroids[l1] + clus_cfg.par_shapes[l1][1]
    else:
        xd = clus_cfg._centroids[l1] - clus_cfg.par_shapes[l1][1]
    return xd


def compute_d(clus_cfg, data, indexes_overLap, l2, xd, sign):
    if sign == "+":
        d_ = (
            data[indexes_overLap[l2]]
            + xd
            + clus_cfg.par_shapes[l2][1]
            - clus_cfg._centroids[l2]
        )
    else:
        d_ = (
            data[indexes_overLap[l2]]
            + xd
            - clus_cfg.par_shapes[l2][1]
            - clus_cfg._centroids[l2]
        )
    return d_


def common_code(clus_cfg, cluster, samples, label, data, d_, indexes_overLap, l1):
    if clus_cfg.par_shapes[l1][0] is not None:
        d_ = clus_cfg.par_shapes[l1][0](
            d_,
            clus_cfg._centroids[l1],
            clus_cfg.par_shapes[l1][1],
            "+",
        )
        if clus_cfg.shapes[l1] is not None:
            d_ = verify_shape(
                d_,
                clus_cfg._shapes[l1],
                clus_cfg.parametres_shapes[l1],
                cluster,
                samples,
                clus_cfg,
                label,
            )
        result = [
            point
            for point in d_
            if np.any(np.array(point) < np.amax(data[indexes_overLap[l1]]))
            and np.any(np.array(point) < np.amax(data[indexes_overLap[l1]]))
        ]
        P = len(result) / samples
        return result, P


def v_sup_per(clus_cfg, xd, l1, percent, P):
    v_sup_m = np.all(
        np.array(xd) >= np.array(clus_cfg._centroids[l1] - clus_cfg.par_shapes[l1][1])
    )

    v_inf_p = np.all(
        np.array(xd) <= np.array(clus_cfg._centroids[l1] + clus_cfg.par_shapes[l1][1])
    )
    per = round(percent, 2) != round(P, 2)
    return v_sup_m, v_inf_p, per


def compute_df(clus_cfg, data, indexes_overLap, l2, xdmin, sign):
    if sign == "+":
        d_f = (
            data[indexes_overLap[l2]]
            + xdmin
            + clus_cfg.par_shapes[l2][1]
            - clus_cfg._centroids[l2]
        )
    else:
        d_f = (
            data[indexes_overLap[l2]]
            + xdmin
            - clus_cfg.par_shapes[l2][1]
            - clus_cfg._centroids[l2]
        )

    return d_f


def cdt(P, percent, result, xd):
    if np.abs(P - percent) < np.abs(p_opt - percent):
        xdmin = xd
        p_opt = P
    return xdmin


def apply_chev_cdt(
    clus_cfg,
    data,
    cluster,
    samples,
    label,
    indexes_overLap,
    percent,
    v_inf_p,
    v_sup_m,
    per,
    l1,
    l2,
    sign,
):
    sign2 = sign
    xd = compute_xd(clus_cfg, l1, sign)
    d_ = compute_d(clus_cfg, data, indexes_overLap, l2, xd, sign)
    result, P = (clus_cfg, cluster, samples, label, data, indexes_overLap, l1)
    xdmin = xd
    p_opt = P
    nb_opt = len(result)
    if P > percent:
        while v_inf_p and v_sup_m and per:
            xd, result, xdmin = fact_apply_chev(
                clus_cfg,
                cluster,
                samples,
                label,
                data,
                indexes_overLap,
                percent,
                l1,
                l2,
                sign2,
                sign,
            )
            v_sup_m, v_inf_p, per = v_sup_per(clus_cfg, xd, l1, percent, P)
        d_f, clus_cfg.chevauchement = common_code2(
            clus_cfg, nb_opt, p_opt, percent, l1, l2, sign
        )
    else:
        while (percent > P) and v_inf_p and v_sup_m and per:
            if sign == "+":
                sign2 = "-"
            if sign == "-":
                sign2 = "+"
            xd, result, xdmin = fact_apply_chev(
                clus_cfg,
                cluster,
                samples,
                label,
                data,
                indexes_overLap,
                percent,
                l1,
                l2,
                sign2,
                sign,
            )
            v_sup_m, v_inf_p, per = v_sup_per(clus_cfg, xd, l1, percent, P)
        d_f, clus_cfg.chevauchement = common_code2(
            clus_cfg, nb_opt, p_opt, percent, l1, l2, sign
        )
    return clus_cfg, nb_opt, p_opt, v_inf_p, v_sup_m, per, xdmin


def update_data_chev(
    clus_cfg, data, indexes_overLap, xdmin, nb_opt, p_opt, percent, l1, l2, sign
):
    d_f = compute_df(clus_cfg, data, indexes_overLap, l2, xdmin, sign)
    clus_cfg.chevauchement = clus_cfg.chevauchement[1:]
    data[indexes_overLap[l2]] = d_f
    loggings_overlap(l1, l2, nb_opt, p_opt)
    if p_opt < percent:
        logging.warning(
            "\n Warning : the percentage of overlap achieved is less than the desired percentage"
        )
    return data, clus_cfg.chevauchement


def apply_overlap(clus_cfg, indexes_overLap, data, cluster, samples, label):
    for i, (l1, l2, percent) in enumerate(clus_cfg.chevauchement):
        if l1 in indexes_overLap.keys() and l2 in indexes_overLap.keys():
            if np.any(clus_cfg._centroids[l1] < clus_cfg._centroids[l2]):
                (
                    clus_cfg,
                    nb_opt,
                    p_opt,
                    v_inf_p,
                    v_sup_m,
                    per,
                    xdmin,
                ) = apply_chev_cdt(
                    clus_cfg,
                    data,
                    cluster,
                    samples,
                    label,
                    indexes_overLap,
                    percent,
                    v_inf_p,
                    v_sup_m,
                    per,
                    l1,
                    l2,
                    "+",
                )
                data, clus_cfg.chevauchement = update_data_chev(
                    clus_cfg,
                    data,
                    indexes_overLap,
                    xdmin,
                    nb_opt,
                    p_opt,
                    percent,
                    l1,
                    l2,
                    "+",
                )
            else:
                (
                    clus_cfg,
                    nb_opt,
                    p_opt,
                    v_inf_p,
                    v_sup_m,
                    per,
                    xdmin,
                ) = apply_chev_cdt(
                    clus_cfg,
                    data,
                    cluster,
                    samples,
                    label,
                    indexes_overLap,
                    percent,
                    v_inf_p,
                    v_sup_m,
                    per,
                    l1,
                    l2,
                    "-",
                )
                data, clus_cfg.chevauchement = update_data_chev(
                    clus_cfg,
                    data,
                    indexes_overLap,
                    xdmin,
                    nb_opt,
                    p_opt,
                    percent,
                    l1,
                    l2,
                    "-",
                )


def loggings_overlap(l1, l2, nb_opt, p_opt):
    logging.info(f"--------------- Overlapp between {l1} & {l2} ------")
    logging.info(
        f"Optimal percent = {p_opt} corresponding to {nb_opt} points on the overlap"
    )
