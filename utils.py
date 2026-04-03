def cal_iou(bbx, bbx_gt):
    """Calculate IoU between two bounding boxes"""
    x1, x2, y1, y2 = bbx
    X1, X2, Y1, Y2 = bbx_gt
    s1 = (y2-y1)*(x2-x1)
    s2 = (Y2-Y1)*(X2-X1)
    jx1 = max(x1, X1)
    jx2 = min(x2, X2)
    jy1 = max(y1, Y1)
    jy2 = min(y2, Y2)
    if jx2 > jx1 and jy2 > jy1:
        s3 = (jx2-jx1)*(jy2-jy1)
    else:
        s3 = 0.0
    return s3/(s1+s2-s3)



def get_center(bbx):
    cx = (bbx[0] + bbx[1]) / 2.0
    cy = (bbx[2] + bbx[3]) / 2.0
    return (cx, cy)


def compute_iou_gradient(bbx, bbx_gt, epsilon=1.0):
    cx, cy = get_center(bbx)
    width = bbx[1] - bbx[0]
    height = bbx[3] - bbx[2]
    
    bbx_cx_plus = [cx - width/2 + epsilon, cx + width/2 + epsilon, bbx[2], bbx[3]]
    bbx_cx_minus = [cx - width/2 - epsilon, cx + width/2 - epsilon, bbx[2], bbx[3]]
    grad_cx = (cal_iou(bbx_cx_plus, bbx_gt) - cal_iou(bbx_cx_minus, bbx_gt)) / (2 * epsilon)
    
    bbx_cy_plus = [bbx[0], bbx[1], cy - height/2 + epsilon, cy + height/2 + epsilon]
    bbx_cy_minus = [bbx[0], bbx[1], cy - height/2 - epsilon, cy + height/2 - epsilon]
    grad_cy = (cal_iou(bbx_cy_plus, bbx_gt) - cal_iou(bbx_cy_minus, bbx_gt)) / (2 * epsilon)
    
    return (grad_cx, grad_cy)


def reward_func_gfirs(bbx, new_bbx, bbx_gt, action, step, prev_bbx=None, 
                      alpha_max=1.0, beta=1.5, lambda_physics=0.3, lambda_efficiency=0.05):
    """GFIRS reward function with PINN-based physics-informed constraints
    Args:
        bbx: Current bounding box
        new_bbx: New bounding box after action
        bbx_gt: Ground truth bounding box
        action: Selected action
        step: Current step
        prev_bbx: Previous bounding box (for physics constraint)
        alpha_max: Maximum step size coefficient
        beta: Step size decay exponent
        lambda_physics: Weight for physics constraint
        lambda_efficiency: Weight for motion efficiency
    Returns:
        Total reward combining task, physics, and efficiency rewards
    """
    new_iou = cal_iou(new_bbx, bbx_gt)
    
    if action == 8:
        if new_iou >= 0.9:
            return 3 + new_iou
        else:
            return -3 - new_iou
    else:
        old_iou = cal_iou(bbx, bbx_gt)
        delta_iou = new_iou - old_iou
        
        if delta_iou > 0:
            r_task = 1 + delta_iou + 0.01 * step
        else:
            r_task = -1 - abs(delta_iou) - 0.01 * step
        
        if prev_bbx is not None:
            c_new = get_center(new_bbx)
            c_old = get_center(bbx)
            c_prev = get_center(prev_bbx)
            
            v_actual = (c_new[0] - c_old[0], c_new[1] - c_old[1])
            
            alpha = alpha_max * ((1 - new_iou) ** beta)
            grad_iou = compute_iou_gradient(new_bbx, bbx_gt)
            v_theory = (-alpha * grad_iou[0], -alpha * grad_iou[1])
            
            l_pde = (v_actual[0] - v_theory[0])**2 + (v_actual[1] - v_theory[1])**2
            r_physics = -l_pde
            
            r_efficiency = -(v_actual[0]**2 + v_actual[1]**2)
            
            r_total = r_task + lambda_physics * r_physics + lambda_efficiency * r_efficiency
        else:
            r_total = r_task
        
        return r_total


def reward_func(bbx, new_bbx, bbx_gt, action, step):
    if action == 8:
       if cal_iou(new_bbx, bbx_gt) >= 0.9:
            return 3+cal_iou(new_bbx, bbx_gt)
       else:
            return -3-cal_iou(new_bbx, bbx_gt)
    else:
       old_iou = cal_iou(bbx, bbx_gt)
       new_iou = cal_iou(new_bbx, bbx_gt)
       if new_iou > old_iou:
           return 1+(new_iou-old_iou)+0.01*step
       elif new_iou <= old_iou:
           return -1-(old_iou-new_iou)-0.01*step