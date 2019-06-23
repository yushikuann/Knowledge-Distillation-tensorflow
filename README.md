# Knowledge-Distillation-tensorflow
if you want train student-net only,change:

student_train_step = tf.train.GradientDescentOptimizer(0.1).minimize(student_loss, var_list=var_student)

as:

student_train_step = tf.train.GradientDescentOptimizer(0.1).minimize(student_loss1, var_list=var_student).

run 
python3 teacher_st_slim.py

make sure the path is right.
 
