# loputoo
Koodid on loodud bakalaureusetöö "Müüontomograafiliste hodoskoopide nurklahutusvõime ja positsioneerimisvõime analüüs" raames. 

Sisendiks võetakse kiudude info ning kasutatakse läbivalt Pandas Dataframe'i nimega event_points_all_df. Näidis selle andmetest:

event	0	1	2	3	4	5	6	7	8	...	10	11	angle resolution	vert3_o_2Dx	vert3_o_2Dy	resolution lines	position resolution	fibers sum	hodoscope angle resolution	hodoscope all angle resolution
0	705304748092516	[[409.5, 200.0], [409.0, 199.25167]]	[[766.5, 198.0], [646.0, 197.25167]]	[[446.5, 100.0], [446.0, 99.25167]]	[[706.5, 98.0], [707.0, 97.25167]]	[[496.5, 0.0]]	[[769.5, -2.0]]	[[199.625835, 409.25]]	[[198.0, 766.5], [197.625835, 645.75]]	[[99.625835, 446.25]]	...	[[0.0, 496.5], [-0.74833, 482.0]]	[[-2.0, 769.5], [-2.74833, 768.0]]	[nan, nan]	19.954479	31.387725	([[nan, nan], [nan, nan]], [[nan, nan], [nan, ...	(nan, nan, nan, nan, nan, nan)	(5, 5)	[0.004556997746996451, nan]	[[0.0017914625112139364, 0.003050772840535934]...

kus event tähistab sündmust, 0 kuni 5 tähistab hodoskoopide numbreid (6 kuni 11 on ebavajalikud), angle resolution on langemisnurga lahutusvõime, vert3_o_2Dx	ja vert3_o_2Dy on xz ja yz ristlõikes langemisnurgad, resolution lines on ekstreemumsirged, position resolution on positsioneerimisvõime, fibers sum on kindla sündmuse ergastunud fiibrite koguarv ning hodoscope angle resolution on hodoskoobisisese hajumisnurga lahutusvõime.
