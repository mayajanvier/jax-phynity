import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from typing import List

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", False)
# Set device to gpu
#jax.config.update('jax_platform_name', 'gpu')

### Physical model Fp
omega0_square_org = 0.2
alpha_org = 0.1 
    
class PendulumParamPDE(eqx.Module):
    omega0_square: jax.Array # type makes it trainable
    alpha: jax.Array 
    is_damped: bool = eqx.field(static=True)  # Static field (not JAX-traceable)
    """ Unified pendulum for generation and inference """

    def __init__(self, is_damped=False, params={"alpha": 0.1, "omega0_square": 0.2}, is_true=False):
        super().__init__()
        self.is_damped = is_damped
        if is_true:
            self.omega0_square = params["omega0_square"] # float will not be trained
            self.alpha = params["alpha"] # float will not be trained
        else:
            self.omega0_square = jnp.array(params["omega0_square"], dtype=jnp.float32) # default unless precised
            if self.is_damped:
                self.alpha = jnp.array(params["alpha"], dtype=jnp.float32) # default unless precised
            else:
                self.alpha = 0.0 # float will not be trained

    def __call__(self, state): 
        # state should be (nc,)
        q, p = state
        dpdt = - self.omega0_square * jnp.sin(q) - self.alpha * p
        return jnp.array([p, dpdt]) 

class Lorenz(eqx.Module) :
    beta : jax.Array
    sigma : jax.Array
    rho : jax.Array

    def __init__(self, beta, sigma, rho) :
        super().__init__()
        self.beta = beta
        self.sigma = sigma
        self.rho = rho

    def __call__(self, t, s, args=None) :
        """
            x, y, z -> dxdt, dydt, dzdt
        """
        x, y, z = s
        dxdt = self.sigma*(y - x )
        dydt = self.rho * x - y - x*z
        dzdt =  x*y - self.beta*z
        return jnp.array([dxdt, dydt, dzdt])

class TwoBody(eqx.Module):
    def __call__(self, t, s):
        x, x_prime, y, y_prime = s
        x_second = -x/ (x**2 + y**2)**(3/2)
        y_second = -y/ (x**2 + y**2)**(3/2)
        return jnp.array([x_prime, x_second, y_prime, y_second])

    
### Data driven model Fa    
class MLP(eqx.Module):
    layers: list # we need to define the type of the attributes of the class in jax

    def __init__(self, key, state_c, hidden, init_gain=0.2):
        super().__init__()
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(state_c, hidden, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(hidden, hidden, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(hidden, state_c, key=key3)]
    
    def __call__(self, x):
        # shape (nc,)
        for layer in self.layers:
            x = layer(x)
        return x

class MLPAngular(eqx.Module):
    layers: list # we need to define the type of the attributes of the class in jax

    def __init__(self, key, state_c, hidden, init_gain=0.2):
        super().__init__()
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(state_c, hidden, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(hidden, hidden, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(hidden, state_c, key=key3)]
    
    def __call__(self, x):
        # shape (nc,)
        # Wrap angles to [0, 2π]
        x = x.at[0].set(jnp.mod(x[0], 2 * jnp.pi)) # theta1
        x = x.at[1].set(jnp.mod(x[1], 2 * jnp.pi)) # theta2
        for layer in self.layers:
            x = layer(x)
        return x


class ConvNetEstimator1D(eqx.Module):
    # from APHYNITY turned into equinox
    layers: list
    def __init__(self, key, state_c=1, hidden=16):
        super().__init__()
        key1, key2, key3 = jax.random.split(key, 3)
        kernel_size = 3
        padding = kernel_size // 2
        self.layers = [
            eqx.nn.Conv1d(state_c, hidden, kernel_size=kernel_size, padding=padding,  use_bias=False, key=key1, padding_mode='CIRCULAR'),
            #eqx.nn.GroupNorm(hidden, momentum=0.9, eps=1e-5),
            eqx.nn.GroupNorm(groups=8, channels=hidden),
            jax.nn.relu,
            eqx.nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=padding, use_bias=False, key=key2,padding_mode='CIRCULAR'),
            #eqx.nn.BatchNorm(hidden, axis_name='batch', momentum=0.9, eps=1e-5),
            eqx.nn.GroupNorm(groups=8, channels=hidden),
            jax.nn.relu,
            eqx.nn.Conv1d(hidden, state_c, kernel_size=kernel_size, padding=padding, use_bias=True, key=key3,padding_mode='CIRCULAR'),
        ]

    def __call__(self, x):
        x = x[None, :]
        for layer in self.layers:
            x = layer(x)
        return x[0, :]


class ConvNetEstimator2D(eqx.Module):
    # CNODE from DIno: four two-dimensional convolutional layers with 64 hidden features, 
    # ReLU activations, 3 ×3 kernel and zero padding
    layers: list
    def __init__(self, key, in_channels=1, out_channels=1, hidden=64):
        super().__init__()
        key1, key2, key3, key4 = jax.random.split(key, 4)
        kernel_size = 3
        padding = kernel_size // 2
        self.layers = [
            eqx.nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, padding=padding,  use_bias=False, key=key1), # padding_mode='CIRCULAR'),
            eqx.nn.GroupNorm(groups=8, channels=hidden),
            jax.nn.relu,
            eqx.nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=padding, use_bias=False, key=key2), #padding_mode='CIRCULAR'),
            eqx.nn.GroupNorm(groups=8, channels=hidden),
            jax.nn.relu,
            eqx.nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=padding, use_bias=False, key=key3,), #padding_mode='CIRCULAR'),
            eqx.nn.GroupNorm(groups=8, channels=hidden),
            jax.nn.relu,
            eqx.nn.Conv2d(hidden, out_channels, kernel_size=kernel_size, padding=padding, use_bias=True, key=key4,), #padding_mode='CIRCULAR'),
        ]

    def __call__(self, x):
        x = x[None, ...]
        for layer in self.layers:
            x = layer(x)
        return x[0, ...]


# Unet2D inspired from PDE-Bench
def conv_block(in_ch, out_ch, key):
    k1, k2 = jax.random.split(key)
    return [
        eqx.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, use_bias=False, key=k1),#, padding_mode='CIRCULAR'),
        #eqx.nn.BatchNorm(out_ch, axis_name="batch"),
        eqx.nn.GroupNorm(groups=8, channels=out_ch),
        jax.nn.tanh,
        eqx.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, use_bias=False, key=k2), #padding_mode='CIRCULAR'),
        #eqx.nn.BatchNorm(out_ch, axis_name="batch"),
        eqx.nn.GroupNorm(groups=8, channels=out_ch),
        jax.nn.tanh,
    ]

def apply_block(block, x):
    for layer in block:
        if isinstance(layer, eqx.nn.BatchNorm):
            x = layer(x, axis_name="batch")
        x = layer(x)
    return x

class UNet2D(eqx.Module):
    # Encoder
    enc1: List
    enc2: List
    enc3: List
    enc4: List

    # Bottleneck
    bottleneck: List

    # Decoder
    dec4: List
    dec3: List
    dec2: List
    dec1: List

    # Convs
    up4: eqx.nn.ConvTranspose2d
    up3: eqx.nn.ConvTranspose2d
    up2: eqx.nn.ConvTranspose2d
    up1: eqx.nn.ConvTranspose2d

    final_conv: eqx.nn.Conv2d

    def __init__(self, in_channels=3, out_channels=1, init_features=32, key=None):
        keys = jax.random.split(key, 20)
        f = init_features

        # Encoder
        self.enc1 = conv_block(in_channels, f, keys[0])
        self.enc2 = conv_block(f, f * 2, keys[1])
        self.enc3 = conv_block(f * 2, f * 4, keys[2])
        self.enc4 = conv_block(f * 4, f * 8, keys[3])

        # Bottleneck
        self.bottleneck = conv_block(f * 8, f * 16, keys[4])

        # Upsampling
        self.up4 = eqx.nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2, key=keys[5])
        self.up3 = eqx.nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2, key=keys[6])
        self.up2 = eqx.nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2, key=keys[7])
        self.up1 = eqx.nn.ConvTranspose2d(f * 2, f, 2, stride=2, key=keys[8])

        # Decoder
        self.dec4 = conv_block(f * 16, f * 8, keys[9])
        self.dec3 = conv_block(f * 8, f * 4, keys[10])
        self.dec2 = conv_block(f * 4, f * 2, keys[11])
        self.dec1 = conv_block(f * 2, f, keys[12])

        # Final
        self.final_conv = eqx.nn.Conv2d(f, out_channels, kernel_size=1, key=keys[13])#,padding_mode='CIRCULAR')

    def maxpool(self, x):
        return eqx.nn.MaxPool2d(2, stride=2)(x)

    def __call__(self, x):
        x = x[None, ...]  # add channel dimension
        e1 = apply_block(self.enc1, x)
        e2 = apply_block(self.enc2, self.maxpool(e1))
        e3 = apply_block(self.enc3, self.maxpool(e2))
        e4 = apply_block(self.enc4, self.maxpool(e3))

        # Bottleneck
        b = apply_block(self.bottleneck, self.maxpool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = jnp.concatenate([d4, e4], axis=0)
        d4 = apply_block(self.dec4, d4)

        d3 = self.up3(d4)
        d3 = jnp.concatenate([d3, e3], axis=0)
        d3 = apply_block(self.dec3, d3)

        d2 = self.up2(d3)
        d2 = jnp.concatenate([d2, e2], axis=0)
        d2 = apply_block(self.dec2, d2)

        d1 = self.up1(d2)
        d1 = jnp.concatenate([d1, e1], axis=0)
        d1 = apply_block(self.dec1, d1)

        return self.final_conv(d1)[0]

# Unet inspired from PDE-Refiner 
class ResBlock1D(eqx.Module):
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm

    def __init__(self, channels, *, key):
        k1, k2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, key=k1
        )
        self.conv2 = eqx.nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, key=k2
        )
        self.norm1 = eqx.nn.GroupNorm(groups=8, channels=channels)
        self.norm2 = eqx.nn.GroupNorm(groups=8, channels=channels)

    def __call__(self, x):
        h = self.conv1(jax.nn.gelu(self.norm1(x)))
        h = self.conv2(jax.nn.gelu(self.norm2(h)))
        return x + h


class Downsample1D(eqx.Module):
    conv: eqx.nn.Conv1d

    def __init__(self, in_ch, out_ch, *, key):
        self.conv = eqx.nn.Conv1d(
            in_ch, out_ch, kernel_size=3, stride=2, padding=1, key=key
        )

    def __call__(self, x):
        return self.conv(x)


class Upsample1D(eqx.Module):
    conv: eqx.nn.ConvTranspose1d

    def __init__(self, in_ch, out_ch, *, key):
        self.conv = eqx.nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1, key=key
        )

    def __call__(self, x):
        return self.conv(x)

class UNet1D(eqx.Module):
    # Encoder
    conv_in: eqx.nn.Conv1d
    rb1_1: ResBlock1D
    rb1_2: ResBlock1D
    down1: Downsample1D

    rb2_1: ResBlock1D
    rb2_2: ResBlock1D
    down2: Downsample1D

    rb3_1: ResBlock1D
    rb3_2: ResBlock1D
    down3: Downsample1D

    rb4_1: ResBlock1D
    rb4_2: ResBlock1D

    # Middle
    mid1: ResBlock1D
    mid2: ResBlock1D

    # Decoder
    up3: Upsample1D
    rb_up3_1: ResBlock1D
    rb_up3_2: ResBlock1D

    up2: Upsample1D
    rb_up2_1: ResBlock1D
    rb_up2_2: ResBlock1D

    up1: Upsample1D
    rb_up1_1: ResBlock1D
    rb_up1_2: ResBlock1D
    rb_up1_3: ResBlock1D

    # Output
    norm_out: eqx.nn.GroupNorm
    conv_out: eqx.nn.Conv1d

    def __init__(self, *, key,c1=32, c2=64, c3=128, c4=256):  #c1=64, c2=128, c3=256, c4=1024):
        keys = jax.random.split(key, 30)
        k = iter(keys)

        # Encoder
        self.conv_in = eqx.nn.Conv1d(1, c1, 3, padding=1, key=next(k))
        self.rb1_1 = ResBlock1D(c1, key=next(k))
        self.rb1_2 = ResBlock1D(c1, key=next(k))
        self.down1 = Downsample1D(c1, c2, key=next(k))

        self.rb2_1 = ResBlock1D(c2, key=next(k))
        self.rb2_2 = ResBlock1D(c2, key=next(k))
        self.down2 = Downsample1D(c2, c3, key=next(k))

        self.rb3_1 = ResBlock1D(c3, key=next(k))
        self.rb3_2 = ResBlock1D(c3, key=next(k))
        self.down3 = Downsample1D(c3, c4, key=next(k))

        self.rb4_1 = ResBlock1D(c4, key=next(k))
        self.rb4_2 = ResBlock1D(c4, key=next(k))

        # Middle
        self.mid1 = ResBlock1D(c4, key=next(k))
        self.mid2 = ResBlock1D(c4, key=next(k))

        # Decoder
        self.up3 = Upsample1D(c4, c3, key=next(k))
        self.rb_up3_1 = ResBlock1D(c3, key=next(k))
        self.rb_up3_2 = ResBlock1D(c3, key=next(k))

        self.up2 = Upsample1D(c3, c2, key=next(k))
        self.rb_up2_1 = ResBlock1D(c2, key=next(k))
        self.rb_up2_2 = ResBlock1D(c2, key=next(k))

        self.up1 = Upsample1D(c2, c1, key=next(k))
        self.rb_up1_1 = ResBlock1D(c1, key=next(k))
        self.rb_up1_2 = ResBlock1D(c1, key=next(k))
        self.rb_up1_3 = ResBlock1D(c1, key=next(k))

        self.norm_out = eqx.nn.GroupNorm(8, c1)
        self.conv_out = eqx.nn.Conv1d(c1, 1, 3, padding=1, key=next(k))

    def __call__(self, x):
        x = x[None, :]  # Add channel dimension
        # Encoder
        x1 = self.conv_in(x)
        x1 = self.rb1_1(x1)
        x1 = self.rb1_2(x1)

        x2 = self.down1(x1)
        x2 = self.rb2_1(x2)
        x2 = self.rb2_2(x2)

        x3 = self.down2(x2)
        x3 = self.rb3_1(x3)
        x3 = self.rb3_2(x3)

        x4 = self.down3(x3)
        x4 = self.rb4_1(x4)
        x4 = self.rb4_2(x4)

        # Middle
        x = self.mid1(x4)
        x = self.mid2(x)

        # Decoder
        x = self.up3(x)
        x = x + x3
        x = self.rb_up3_1(x)
        x = self.rb_up3_2(x)

        x = self.up2(x)
        x = x + x2
        x = self.rb_up2_1(x)
        x = self.rb_up2_2(x)

        x = self.up1(x)
        x = x + x1
        x = self.rb_up1_1(x)
        x = self.rb_up1_2(x)
        x = self.rb_up1_3(x)

        x = self.conv_out(jax.nn.gelu(self.norm_out(x)))
        return x[0, :] # remove channel dimension

if __name__ == '__main__':
    nb_neurons = 200
    input = jax.random.normal(jax.random.PRNGKey(0), (25,2,40)) # batch, state, time
    print(input.shape)
    model_aug = MLP(jax.random.PRNGKey(0), 2, nb_neurons)
    print(model_aug)
    output = jax.vmap(model_aug)(input)
    print(output.shape)

    model_phy = PendulumParamPDE(is_complete=True, real_params=None) 
    print(model_phy)
    state = jax.random.normal(jax.random.PRNGKey(0), (1,2,3))
    out = model_phy(state)
    print(out.shape) # same shape as input
    print("params", model_phy.alpha, model_phy.omega0_square)  
    
