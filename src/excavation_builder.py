class ExcavationBuilder:
    """用于自动构建基坑工程几何模型的构建器。"""
    def __init__(self, g_i):
        """
        :param g_i: PLAXIS 提供的 Geometry 接口 (global object)，用于进行建模命令调用。
        """
        self.g_i = g_i
    
    def build_excavation_model(self):
        """按照预定义参数构建基坑开挖模型的几何体（不含计算阶段）。"""
        g_i = self.g_i  # 简化引用
        
        # 1. 定义模型边界尺寸（如矩形场地范围）
        # 例如：水平尺寸 40m x 40m，假设地面标高 z=0
        length_x = 40.0
        length_y = 40.0
        g_i.SoilContour.initializerectangular(0.0, 0.0, length_x, length_y)
        
        # 2. 定义土层结构（通过钻孔和土层厚度）
        # 放置一个钻孔用于定义全场土层，钻孔位置取场地中央
        borehole_x = length_x / 2.0
        borehole_y = length_y / 2.0
        g_i.borehole(borehole_x, borehole_y)
        # 定义土层厚度，例如第一层厚度10m（基坑开挖深度）
        first_layer_thickness = 10.0
        g_i.soillayer(first_layer_thickness)  # 增加一个土层界面：顶层厚度10m
        # 此时模型中应有两层土：0~-10m 和 -10m以下第二层
        
        # （可选）为土层分配土性材料，此处略过，假定使用默认土性以关注几何。
        
        # 3. 绘制基坑开挖轮廓（在地面上画出开挖区域多边形）
        # 例如：基坑为 20m x 20m 的方形，位于场地中央，深度10m
        pit_length_x = 20.0
        pit_length_y = 20.0
        pit_depth = 10.0
        # 计算开挖矩形在场地中的坐标范围（中心对齐）
        pit_x0 = (length_x - pit_length_x) / 2.0  # 开挖区域左下角 x
        pit_y0 = (length_y - pit_length_y) / 2.0  # 开挖区域左下角 y
        pit_x1 = pit_x0 + pit_length_x           # 开挖区域右上角 x
        pit_y1 = pit_y0 + pit_length_y           # 开挖区域右上角 y
        # 创建表示开挖区域的地表多边形
        excavation_polygon = g_i.polygon(
            (pit_x0, pit_y0, 0.0),
            (pit_x1, pit_y0, 0.0),
            (pit_x1, pit_y1, 0.0),
            (pit_x0, pit_y1, 0.0)
        )
        
        # 4. 将开挖区域多边形向下拉伸（挤出）至基坑底部深度，以形成开挖体积
        old_surfaces = set(g_i.Surfaces[:])  # 挤出前现有表面集合，供后续比对新生成的表面
        # 沿负Z方向挤出多边形形成体积（深度 pit_depth）
        result = g_i.extrude((excavation_polygon), 0.0, 0.0, -pit_depth)
        # 挤出后会生成新的土体单元（基坑内部土体）以及相应表面
        # 删除原地面的开挖轮廓多边形对象，因为它已被挤出，不再需要
        g_i.delete(excavation_polygon)
        # 提取挤出结果：新生成的体积和底面多边形
        new_soil_volume = result[0]      # 新的土体对象（基坑内土体块）
        bottom_polygon = result[1]       # 基坑底面多边形表面
        
        # 5. 绘制围护结构（基坑四周围墙）
        # 首先创建板单元材料（如混凝土板），用于赋予围墙结构属性
        g_i.platemat("Identification", "Wall")  # 创建名为 "Wall" 的板材质
        wall_material = g_i.Wall  # 获取刚创建的板材质对象
        
        # 找出新生成的侧壁表面（即基坑四周竖直面），将其赋予板材质
        new_surfaces = set(g_i.Surfaces[:])
        # 新增表面 = 挤出后表面集合 与 挤出前表面集合的差集
        added_surfaces = new_surfaces - old_surfaces
        # 去除基坑底部水平面，只保留竖直的侧壁面集合
        if bottom_polygon in added_surfaces:
            added_surfaces.remove(bottom_polygon)
        side_wall_surfaces = added_surfaces  # 剩下的即四周围护墙面
        # 将每个围护墙面指定为板结构（赋予板材质）
        for wall_surface in side_wall_surfaces:
            g_i.setmaterial(wall_surface, wall_material)
        
        # （可选）打开围护墙正/反面的接口，如有需要:
        # for wall_surface in side_wall_surfaces:
        #     g_i.posinterface(wall_surface)
        #     g_i.neginterface(wall_surface)
        
        # 模型几何构建完成。可以返回关键对象供进一步检查（测试或后续操作）。
        return {
            "new_volume": new_soil_volume,
            "bottom_polygon": bottom_polygon,
            "wall_surfaces": side_wall_surfaces
        }
