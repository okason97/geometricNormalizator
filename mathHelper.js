const centroid = (points) => {
  var x = 0, y = 0
  points.forEach(point => {
    x += point.x
    y += point.y
  })
  return {x: x/points.length, y: y/points.length}
}

const slope = (pointA, pointB) => {
  return (pointB.y-pointA.y)/(pointB.x-pointA.x)
}

const detectionCoordinates = (detection) =>{
  return [
    {x: detection.x, y: detection.y},
    {x: detection.x + detection.width, y: detection.y},
    {x: detection.x + detection.width, y: detection.y + detection.height},
    {x: detection.x, y: detection.y + detection.height},
  ]
}

const rotate = (cx, cy, x, y, angle) => {
  var cos = Math.cos(angle),
    sin = Math.sin(angle),
    nx = (cos * (x - cx)) + (sin * (y - cy)) + cx,
    ny = (cos * (y - cy)) - (sin * (x - cx)) + cy;
  return {x: nx, y: ny};
}

const distance = (p1, p2) => {
  return Math.hypot(p2.x-p1.x, p2.y-p1.y)
}

module.exports = {
  rotate: rotate,
  distance: distance,
  centroid: centroid,
  slope: slope,
  detectionCoordinates: detectionCoordinates
};