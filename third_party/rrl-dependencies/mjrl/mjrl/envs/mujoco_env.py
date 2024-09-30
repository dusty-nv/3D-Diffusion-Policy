import os
import PIL

from gym import error, spaces
from gym.utils import seeding
from collections import namedtuple

import numpy as np
from os import path
import gym
import six
import time as timer

try:
    import mujoco
    #from mujoco import load_model_from_path, MjSim, MjViewer
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

def get_sim(model_path):
    if model_path.startswith("/"):
        fullpath = model_path
    else:
        fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
    if not path.exists(fullpath):
        raise IOError("File %s does not exist" % fullpath)
    model = mujoco.MjModel.from_xml_path(fullpath)
    data = mujoco.MjData(model)
    return model, data #MjSim(model)

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path=None, frame_skip=1, sim=None):

        if sim is None:
            self.model, self.data = get_sim(model_path)
        else:
            self.model, self.data = sim.model, sim.data

        self.sim = namedtuple('MjSim', ['model', 'data', 'env'])(
            model=self.model, data=self.data, env=self,
        )
        
        self.render_egl = None
        self.render_ctx = None
        self.render_cam = None
        self.render_opt = None
        self.render_dim = (-1,-1)
        
        self.scene = None
        
        self.frame_count = 0
        self.frame_skip = frame_skip
        
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.mujoco_render_frames = False

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        try:
            observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        except NotImplementedError:
            observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high, dtype=np.float32)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def mj_viewer_setup(self):
        """
        Due to specifics of new mujoco rendering, the standard viewer cannot be used
        with this set-up. Instead we use this mujoco specific function.
        """
        pass

    def viewer_setup(self):
        """
        Does not work. Use mj_viewer_setup() instead
        """
        pass

    def evaluate_success(self, paths, logger=None):
        """
        Log various success metrics calculated based on input paths into the logger
        """
        pass

    # -----------------------------

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        for i in range(self.model.nu):
            self.data.ctrl[i] = ctrl[i]
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)#self.sim.step()
            if self.mujoco_render_frames is True:
                self.mj_render()

    def mj_render(self):
        try:
            self.viewer.render()
        except:
            self.mj_viewer_setup()
            self.viewer._run_speed = 0.5
            #self.viewer._run_speed /= self.frame_skip
            self.viewer.render()

    def render(self, width=640, height=480, mode='offscreen', camera_name='agentview', depth=False, device_id=0, **kwargs):
        if mode != 'offscreen':
            raise NotImplementedError(f"only offscreen rendering is currently supported (requested '{mode}')")
         
        if kwargs:
            print(f"Warning - unused kwargs in mujoco_env.render()  {list(kwargs.keys())}")
           
        render_dim = (width, height)
        
        if self.render_dim != render_dim:
            print(f"Creating EGL context {render_dim}  (previous {self.render_dim})")
            
            self.render_egl = mujoco.GLContext(max_width=render_dim[1], max_height=render_dim[0])
            self.render_egl.make_current()
            self.render_cam = mujoco.MjvCamera()
            self.render_opt = mujoco.MjvOption()
            self.render_dim = render_dim

            mujoco.mjv_defaultCamera(self.render_cam)
            mujoco.mjv_defaultOption(self.render_opt)
            
            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self.viewport = mujoco.MjrRect(0, 0, self.render_dim[0], self.render_dim[1])
            self.render_ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        self.render_egl.make_current()
        mujoco.mjv_updateScene(self.model, self.data, self.render_opt, None, self.render_cam, 0, self.scene)
        
        img = np.ones((self.render_dim[1], self.render_dim[0], 3), dtype=np.uint8)
        depth_img = np.zeros((self.render_dim[1], self.render_dim[0], 1), dtype=np.float32) if depth else None
        
        mujoco.mjr_render(self.viewport, self.scene, self.render_ctx)
        mujoco.mjr_readPixels(img, depth_img, self.viewport, self.render_ctx)
        
        dump_dir = os.environ.get('MUJOCO_RENDER_DUMP_DIR')
        
        if dump_dir:
            print(self.render_cam, self.scene, self.render_opt)
            print(img.shape, img.dtype)
            os.makedirs(dump_dir, exist_ok=True)
            PIL.Image.fromarray(img).save(os.path.join(dump_dir, f'{self.frame_count}.jpg'))
            
        self.frame_count += 1  

        return (img, depth_img) if depth else img

    def _get_viewer(self):
        pass
        #return None

    def state_vector(self):
        state = self.sim.get_state()
        return np.concatenate([
            state.qpos.flat, state.qvel.flat])

    # -----------------------------

    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration'):
        self.mujoco_render_frames = True
        for ep in range(num_episodes):
            o = self.reset()
            d = False
            t = 0
            score = 0.0
            while t < horizon and d is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, _ = self.step(a)
                t = t+1
                score = score + r
            print("Episode score = %f" % score)
        self.mujoco_render_frames = False

    def visualize_policy_offscreen(self, policy, horizon=1000,
                                   num_episodes=1,
                                   frame_size=(640,480),
                                   mode='exploration',
                                   save_loc='/tmp/',
                                   filename='newvid',
                                   camera_name=None):
        import skvideo.io
        for ep in range(num_episodes):
            print("Episode %d: rendering offline " % ep, end='', flush=True)
            o = self.reset()
            d = False
            t = 0
            arrs = []
            t0 = timer.time()
            while t < horizon and d is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, _ = self.step(a)
                t = t+1
                curr_frame = self.sim.render(width=frame_size[0], height=frame_size[1],
                                             mode='offscreen', camera_name=camera_name, device_id=0)
                arrs.append(curr_frame[::-1,:,:])
                print(t, end=', ', flush=True)
            file_name = save_loc + filename + str(ep) + ".mp4"
            skvideo.io.vwrite( file_name, np.asarray(arrs))
            print("saved", file_name)
            t1 = timer.time()
            print("time taken = %f"% (t1-t0))
