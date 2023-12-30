import tensorflow as tf

def main():
	string = tf.Variable("this is a string", tf.string)
	print("rank(string): ", tf.rank(string))
	print("string.shape: ", string.shape)

	number = tf.Variable(324, tf.int16)
	print("number.shape: ", number.shape)

	floating = tf.Variable(3.567, tf.float64)
	print("rank(floating): ", tf.rank(floating))

	var1 = tf.Variable(["Test", "ok"], tf.string)
	print("rank(var1): ", tf.rank(var1))
	print("var1.shape: ", var1.shape)

	var2 = tf.Variable([["test", "ok", "4"], ["test", "1", "2"]], tf.string)
	print("rank(var2): ", tf.rank(var2))
	print("var2.shape: ", var2.shape)

	var3 = tf.Variable([["test", "ok", "4", "3"], ["test", "1", "2", "1"]], tf.string)
	print("rank(var3): ", tf.rank(var3))
	print("var3.shape: ", var3.shape)

	var4 = tf.Variable([[["test", "ok", "4", "3"]], [["test", "1", "2", "1"]]], tf.string)
	print("rank(var4): ", tf.rank(var4))
	print("var4.shape: ", var4.shape)

	var5 = tf.Variable([[["test", "ok", "4", "3"], ["test", "1", "2", "1"]]], tf.string)
	print("rank(var5): ", tf.rank(var5))
	print("var5.shape: ", var5.shape)

	var6 = tf.Variable([[[["test", "ok", "4", "3"]], [["test", "1", "2", "1"]]]], tf.string)
	print("rank(var6): ", tf.rank(var6))
	print("var6.shape: ", var6.shape)

	var7 = tf.ones([1, 2, 3])
	print("var7: ", var7)
	print("rank(var7): ", tf.rank(var7))
	print("shape(var7): ", tf.shape(var7))
	print("var7.shape: ", var7.shape)

	var8 = tf.ones([1, 2, 3, 4])
	print("var8: ", var8)
	print("rank(var8): ", tf.rank(var8))
	print("shape(var8): ", tf.shape(var8))
	print("var8.shape: ", var8.shape)

	var9 = tf.reshape(var8, [2, 3, 4, 1])
	print("var9: ", var9)
	print("rank(var9): ", tf.rank(var9))
	print("var9.shape: ", var9.shape)

	var10 = tf.reshape(var9, [3, -1])
	print("var10: ", var10)
	print("rank(var10): ", tf.rank(var10))
	print("var10.shape: ", var10.shape)

	var11 = tf.reshape(var10, [-1, 2])
	print("var11: ", var11)
	print("rank(var11): ", tf.rank(var11))
	print("var11.shape: ", var11.shape)


if __name__ == '__main__':
	main()
