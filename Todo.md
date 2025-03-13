# frictions

I have set anisotropic friction using a directly defined contact (from the documentation, that seems to be the only way, this is also what is suggested here https://github.com/google-deepmind/mujoco/issues/67 ).

However, I'm not sure which of the two friction="a b" coefficients will go to `tangent_1 and tangent_2`. Plotting with matplotlib seems to indicate that the first goes to `tangent_1` and so on,
but plotting contact splist in mujoco's GUI seems to contradict that... I'm not sure how to interpret this. I hope tangent vectors don't flip midway!

solutions: 1) implement PID, if it works correctly, then frictions might be set up correctly
           2) do ask the question on mujoco's git

#PID

UPDATE: what is below is correct, except that I'm switching to torque control for easier modeling via lagrangian mechanics

So, dumb pid control à la 

err=cross(up,gravity)

will produce 

u=K_p*err+K_d*blablabla

which is an increment to the "up" vector, really, NOT the wheel velocity control that we need. This is especially obvious when the gains are scalar. Then, since gravity is align with the z vector,
the err=cross(up,gravity) will lie in the xy plane, meaning that the last component of u will always be zero. So, what a controller with scalar gains would be doing is to tell us how the robot's 
base has to move horizontally to compensate for the change in tilt. If we consider that the gains are 3x3 matrices, we'd be adding a linear transform, but 
 1. I doubt that the resulting u would live in motor command space (this has to be a bit more non-linear)
 2. Tuning those 3x3 matrices sounds like a nightmare

So... I think we need a model of the robot to do this:

   `velocity_control`=F(u) #where u are the horizontal increments 

So yeah, I guess it's time to finally read those papers!

#MISC

In case you haven't touched this code for a while: there are sleep functions in the bbot.py file that are just here for debug!
