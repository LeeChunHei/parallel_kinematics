import torch
import xml.etree.ElementTree as ET

def _quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    one = torch.ones_like(r)
    zero = torch.zeros_like(r)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            zero,
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            zero,
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
            zero,
            zero,zero,zero,one
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (4, 4))

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 4, 4).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = ( one, zero, zero, zero, 
                  zero,  cos, -sin, zero, 
                  zero,  sin,  cos, zero,
                  zero, zero, zero,  one)
    elif axis == "Y":
        R_flat = ( cos, zero,  sin, zero,
                  zero,  one, zero, zero,
                  -sin, zero,  cos, zero,
                  zero, zero, zero,  one)
    elif axis == "Z":
        R_flat = ( cos, -sin, zero, zero,
                   sin,  cos, zero, zero,
                  zero, zero,  one, zero,
                  zero, zero, zero,  one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (4, 4))

class Chain(object):
    def __init__(self, link_list, device="cuda") -> None:
        self.link_list = link_list
        self.device = device

    def __str__(self):
        ret = ""
        for link in self.link_list:
            depth = self._find_depth(link)
            ret += "    "*depth + "<joint:" + link.joint_axis + "> " + link.name + "\r\n"
        return ret

    def _find_depth(self, link):
        if link.parent == None:
            return 0
        else:
            return 1 + self._find_depth(link.parent)

    def get_all_link_name(self):
        ret = []
        for link in self.link_list:
            ret += [link.name]
        return ret

    def get_all_joint_name(self):
        ret = []
        for link in self.link_list:
            ret += link.joint_name
        return ret

    def forward_kinematics(self, joint_pos:torch.Tensor, 
                                 root_rot:torch.Tensor, 
                                 root_pos:torch.Tensor):
        self.link_list[0].add_translation(root_pos)
        self.link_list[0].add_rotation_quad(root_rot)
        dof_offset = 0
        for link in self.link_list:
            dof_len = len(link.joint_axis)
            if dof_len == 0:
                continue
            link.add_rotation(joint_pos[:,dof_offset:dof_offset+dof_len])
            dof_offset += dof_len
        for link in self.link_list:
            link.apply_transformation()
        ret = None
        for link in self.link_list:
            pos = link.get_pos()
            if ret == None:
                ret = pos
            else:
                ret = torch.cat([ret, pos], dim=-1)
        return ret

    def reset(self):
        for link in self.link_list:
            link.reset()

class Link(object):
    def __init__(self, name, pos, parent, joint_name, joint_axis, device) -> None:
        self.name = name
        self.joint_name = joint_name
        self.device = device
        self.link_origin = torch.FloatTensor([[1, 0, 0, pos[0]],
                                              [0, 1, 0, pos[1]],
                                              [0, 0, 1, pos[2]],
                                              [0, 0, 0,      1]]).to(self.device)
        self.result_matrix = None
        self.parent = parent
        self.joint_axis = joint_axis

    def add_rotation(self, angle: torch.Tensor):
        matrices = [
            _axis_angle_rotation(c, e)
            for c, e in zip(self.joint_axis, torch.unbind(angle, -1))
        ]
        r_m = matrices[0]
        for i in range(1,len(matrices)):
            r_m = torch.matmul(r_m, matrices[i])
        if self.result_matrix == None:
            self.result_matrix = r_m
        else:
            self.result_matrix = torch.matmul(self.result_matrix, r_m)

    def add_rotation_quad(self, quad: torch.Tensor):
        r_m = _quaternion_to_matrix(quad)
        if self.result_matrix == None:
            self.result_matrix = r_m
        else:
            self.result_matrix = torch.matmul(self.result_matrix, r_m)

    def add_translation(self, trans: torch.Tensor):
        '''
        expect trans shape = (..., 3)
        '''
        t_m = torch.eye(4).to(self.device).repeat((trans.shape[0],1)).reshape(-1,4,4)
        t_m[:,:3,3] = trans
        if self.result_matrix == None:
            self.result_matrix = t_m
        else:
            self.result_matrix = torch.matmul(self.result_matrix, t_m)

    def apply_transformation(self):
        if self.result_matrix == None:
            self.result_matrix = self.link_origin
        else:
            self.result_matrix = torch.matmul(self.link_origin, self.result_matrix)
        if self.parent != None:
            self.result_matrix = torch.matmul(self.parent.result_matrix, self.result_matrix)

    def get_pos(self):
        # if self.parent != None:
        #     result = torch.matmul(self.parent.result_matrix, self.result_matrix)
        # else:
        result = self.result_matrix
        return result[:,:3,3].reshape(-1,3)
    
    def reset(self):
        self.result_matrix = None

def create_chain_from_mjcf(mjcf_path, device="cuda"):
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    #find the first body tag
    for body in root.iter('body'):
        first_body = body
        break
    
    def create_link_chain(parent, xml_parent, device="cuda"):
        link_pos = [float(p) for p in xml_parent.get('pos').split(' ')]
        joint_axis = ""
        joint_axis_str = ["X", "Y", "Z"]
        joint_name = []
        joint_list = xml_parent.findall('joint')
        for joint in joint_list:
            axis = joint.get('axis').split(' ')
            joint_axis += joint_axis_str[axis.index('1')]
            joint_name.append(joint.get('name'))
        link_list = [Link(name = xml_parent.get('name'),
                          pos = link_pos,
                          parent = parent,
                          joint_name = joint_name,
                          joint_axis = joint_axis,
                          device = device)]
        for body in xml_parent.findall('body'):
            link_list += create_link_chain(link_list[0], body, device)
        return link_list

    link_list = create_link_chain(None, first_body, device)
    chain = Chain(link_list, device)
    return chain
