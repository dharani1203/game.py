from PIL import Image, ImageDraw

def create_shape_image(name, draw_fn):
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))  # Transparent canvas
    draw = ImageDraw.Draw(img)
    draw_fn(draw)
    img.save(f"shape_{name}.png")
    print(f"Saved: shape_{name}.png")

def draw_circle(draw):
    draw.ellipse((10, 10, 90, 90), fill="red", outline="black", width=3)

def draw_square(draw):
    draw.rectangle((10, 10, 90, 90), fill="blue", outline="black", width=3)

def draw_rectangle(draw):
    draw.rectangle((10, 25, 90, 75), fill="green", outline="black", width=3)

def draw_triangle(draw):
    draw.polygon([(50, 10), (90, 90), (10, 90)], fill="orange", outline="black")

# Generate all shapes
create_shape_image("circle", draw_circle)
create_shape_image("square", draw_square)
create_shape_image("rectangle", draw_rectangle)
create_shape_image("triangle", draw_triangle)