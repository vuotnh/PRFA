"""
Implements handy numerical computational functions
"""
import numpy as np
import torch as ch
from torch.nn.modules import Upsample
import torch


def norm(t):
    """
    Return the norm of a tensor (or numpy) along all the dimensions except the first one
    :param t:
    :return:
    """
    _shape = t.shape
    batch_size = _shape[0]
    num_dims = len(_shape[1:])
    if ch.is_tensor(t):
        norm_t = ch.sqrt(t.pow(2).sum(dim=[_ for _ in range(1, len(_shape))])).view([batch_size] + [1] * num_dims)
        norm_t += (norm_t == 0).float() * np.finfo(np.float64).eps
        return norm_t
    else:
        _norm = np.linalg.norm(
            t.reshape([batch_size, -1]), axis=1, keepdims=1
        ).reshape([batch_size] + [1] * num_dims)
        return _norm + (_norm == 0) * np.finfo(np.float64).eps


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def eg_step(x, g, lr):
    """
    Performs an exponentiated gradient step in the convex body [-1,1]
    :param x: batch_size x dim x .. tensor (or numpy) \in [-1,1]
    :param g: batch_size x dim x .. tensor (or numpy)
    :param lr: learning rate (step size)
    :return:
    """
    # from [-1,1] to [0,1]
    real_x = (x + 1.) / 2.
    if ch.is_tensor(x):
        pos = real_x * ch.exp(lr * g)
        neg = (1 - real_x) * ch.exp(-lr * g)
    else:
        pos = real_x * np.exp(lr * g)
        neg = (1 - real_x) * np.exp(-lr * g)
    new_x = pos / (pos + neg)
    return new_x * 2 - 1


def step(x, g, lr):
    """
    Performs a step with no lp-ball constraints
    :param x: batch_size x dim x .. tensor (or numpy)
    :param g: batch_size x dim x .. tensor (or numpy)
    :param lr: learning rate (step size)
    :return:
    """
    return x + lr * g


def lp_step(x, g, lr, p):
    """
    performs lp step of x in the direction of g, where the norm is computed
    across all the dimensions except the first one (assuming it's the batch_size)
    :param x: batch_size x dim x .. tensor (or numpy)
    :param g: batch_size x dim x .. tensor (or numpy)
    :param lr: learning rate (step size)
    :param p: 'inf' or '2'
    :return:
    """
    if p == 'inf':
        return linf_step(x, g, lr)
    elif p == '2':
        return l2_step(x, g, lr)
    else:
        raise Exception('Invalid p value')


def l2_step(x, g, lr):
    """
    performs l2 step of x in the direction of g, where the norm is computed
    across all the dimensions except the first one (assuming it's the batch_size)
    :param x: batch_size x dim x .. tensor (or numpy)
    :param g: batch_size x dim x .. tensor (or numpy)
    :param lr: learning rate (step size)
    :return:
    """
    # print(x.device)
    # print(g.device)
    # print(norm(g).device)
    return x + lr * g / norm(g)


def linf_step(x, g, lr):
    """
    performs linfinity step of x in the direction of g
    :param x: batch_size x dim x .. tensor (or numpy)
    :param g: batch_size x dim x .. tensor (or numpy)
    :param lr: learning rate (step size)
    :return:
    """
    if ch.is_tensor(x):
        return x + lr * ch.sign(g)
    else:
        return x + lr * np.sign(g)


def l2_proj_maker(xs, eps):
    """
    makes an l2 projection function such that new points
    are projected within the eps l2-balls centered around xs
    :param xs:
    :param eps:
    :return:
    """
    if ch.is_tensor(xs):
        orig_xs = xs.clone()

        def proj(new_xs):
            delta = new_xs - orig_xs
            norm_delta = norm(delta)
            if np.isinf(eps):  # unbounded projection
                return orig_xs + delta
            else:
                return orig_xs + (norm_delta <= eps).float() * delta + (
                        norm_delta > eps).float() * eps * delta / norm_delta
    else:
        orig_xs = xs.copy()

        def proj(new_xs):
            delta = new_xs - orig_xs
            norm_delta = norm(delta)
            if np.isinf(eps):  # unbounded projection
                return orig_xs + delta
            else:
                return orig_xs + (norm_delta <= eps) * delta + (norm_delta > eps) * eps * delta / norm_delta

    return proj


def linf_proj_maker(xs, eps):
    """
    makes an linf projection function such that new points
    are projected within the eps linf-balls centered around xs
    :param xs:
    :param eps:
    :return:
    """
    if ch.is_tensor(xs):
        orig_xs = xs.clone()

        def proj(new_xs):
            return orig_xs + ch.clamp(new_xs - orig_xs, - eps, eps)
    else:
        orig_xs = xs.copy()

        def proj(new_xs):
            return np.clip(new_xs, orig_xs - eps, orig_xs + eps)
    return proj


def upsample_maker(target_h, target_w):
    """
    makes an upsampler which takes a numpy tensor of the form
    minibatch x channels x h x w and casts to
    minibatch x channels x target_h x target_w
    :param target_h: int to specify the desired height
    :param target_w: int to specify the desired width
    :return:
    """
    _upsampler = Upsample(size=(target_h, target_w))

    def upsample_fct(xs):
        if ch.is_tensor(xs):
            return _upsampler(xs)
        else:
            return _upsampler(ch.from_numpy(xs)).numpy()

    return upsample_fct


def hamming_dist(a, b):
    """
    reurns the hamming distance of a to b
    assumes a and b are in {+1, -1}
    :param a:
    :param b:
    :return:
    """
    assert np.all(np.abs(a) == 1.), "a should be in {+1,-1}"
    assert np.all(np.abs(b) == 1.), "b should be in {+1,-1}"
    return sum([_a != _b for _a, _b in zip(a, b)])


# def tf_nsign(t):
#     """
#     implements a custom non-standard sign operation in tensor flow
#     where sing(t) = 1 if t == 0
#     :param t:
#     :return:
#     """
#     return tf.sign(tf.sign(t) + 0.5)


def sign(t, is_ns_sign=True):
    """
    Given a tensor t of `batch_size x dim` return the (non)standard sign of `t`
    based on the `is_ns_sign` flag
    :param t: tensor of `batch_size x dim`
    :param is_ns_sign: if True uses the non-standard sign function
    :return:
    """
    _sign_t = ch.sign(t) if ch.is_tensor(t) else np.sign(t)
    if is_ns_sign:
        _sign_t[_sign_t == 0.] = 1.
    return _sign_t


def noisy_sign(t, retain_p=1, crit='top', is_ns_sign=True):
    """
    returns a noisy version of the tensor `t` where
    only `retain_p` * 100 % of the coordinates retain their sign according
    to a `crit`.
    The noise is of the following effect
        sign(t) * x where x \in {+1, -1}
    Thus, if sign(t) = 0, sign(t) * x is always 0 (in case of `is_ns_sign=False`)
    :param t: tensor of `batch_size x dim`
    :param retain_p: fraction of coordinates
    :param is_ns_sign: if True uses  the non-standard sign function
    :return:
    """
    assert 0. <= retain_p <= 1., "retain_p value should be in [0,1]"

    _shape = t.shape
    t = t.reshape(_shape[0], -1)
    batch_size, dim = t.shape

    sign_t = sign(t, is_ns_sign=is_ns_sign)
    k = int(retain_p * dim)

    if k == 0:  # noise-ify all
        return (sign_t * np.sign((np.random.rand(batch_size, dim) < 0.5) - 0.5)).reshape(_shape)
    if k == dim:  # retain all
        return sign_t.reshape(_shape)

    # do topk otheriwise
    noisy_sign_t = sign_t * np.sign((np.random.rand(*t.shape) < 0.5) - 0.5)
    _rows = np.zeros((batch_size, k), dtype=np.intp) + np.arange(batch_size)[:, None]
    if crit == 'top':
        _temp = np.abs(t)
    elif crit == 'random':
        _temp = np.random.rand(*t.shape)
    else:
        raise Exception('Unknown criterion for topk')

    _cols = np.argpartition(_temp, -k, axis=1)[:, -k:]
    noisy_sign_t[_rows, _cols] = sign_t[_rows, _cols]
    return noisy_sign_t.reshape(_shape)


# def lp_step_patch():

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 0.0000]])
        >>> bbox_overlaps(bboxes1, bboxes2, mode='giou', eps=1e-7)
        tensor([[0.5000, 0.0000, -0.5000],
                [-0.2500, -0.0500, 1.0000],
                [-0.8371, -0.8766, -0.8214]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


if __name__ == '__main__':
    print("I am just a module to be imported by others; testing here")
    t = ch.randn(2, 2, 2)
    t_np = t.numpy()

    tt = ch.randn(2, 2, 2)
    tt_np = tt.numpy()

    linf_proj_t = linf_proj_maker(t, 0.4)
    linf_proj_np = linf_proj_maker(t_np, 0.4)
    l2_proj_t = l2_proj_maker(t, 0.4)
    l2_proj_np = l2_proj_maker(t_np, 0.4)

    assert np.allclose(norm(t).numpy(), norm(t_np))
    assert np.allclose(linf_step(t, tt, 0.2), linf_step(t_np, tt_np, 0.2))
    assert np.allclose(l2_step(t, tt, 0.2), l2_step(t_np, tt_np, 0.2))
    assert np.allclose(step(t, tt, 0.2), step(t_np, tt_np, 0.2))
    assert np.allclose(linf_proj_t(tt), linf_proj_np(tt_np))
    assert np.allclose(l2_proj_t(tt), l2_proj_np(tt_np))
