import math
import matplotlib.pyplot as plt



def main():
	alpha = 82.6 / 180 * math.pi
	beta  = 73.5 / 180 * math.pi

	tg_a = math.tan(alpha)
	tg_b = math.tan(beta)

	l = 1.86
	d = 3.6

	delta = (tg_b * l / 2 - d) / (tg_b - tg_a)
	h = delta * tg_a

	upper_point        = (0.0, d)
	middle_point_left  = (-l / 2 + delta, h)
	middle_point_right = (+l / 2 - delta, h)
	lower_point_left   = (-l / 2, 0.0)
	lower_point_right  = (+l / 2, 0.0)

	points = [
		upper_point,
		middle_point_left,
		lower_point_left,
		lower_point_right,
		middle_point_right,
		upper_point
	]

	for x, y in points:
		print(x, ' ', y)

	x = [x for x, y in points]
	y = [y for x, y in points]

	plt.plot(x, y)
	plt.show()
	

if __name__ == "__main__":
	main()
