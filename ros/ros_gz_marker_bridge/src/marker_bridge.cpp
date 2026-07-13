// Relay visualization_msgs/MarkerArray -> Gazebo GUI markers.
//
// Markers in Gazebo are service-only (the GUI MarkerManager advertises
// /marker + /marker_array; there is no inbound marker topic to bridge with
// ros_gz_bridge). Driving that service from the `gz service` CLI re-pays
// process start + transport discovery every call (~450 ms) -> ~2 Hz. This node
// holds ONE long-lived gz::transport::Node (discovery once) and batches each
// incoming MarkerArray into a single gz::msgs::Marker_V request, so updates run
// at the publisher's rate (tens of Hz) with no per-call cost.
//
// Geometry stays single-source in the (Python) publisher: it emits standard
// RViz markers already in the Gazebo world frame (ENU, z-up); this node is a
// dumb, stable field converter. The same topic also renders in RViz2.
// header.frame_id is ignored (world frame assumed; no TF lookup).

#include <cstdlib>
#include <functional>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/marker.pb.h>
#include <gz/msgs/marker_v.pb.h>
#include <gz/transport/Node.hh>

namespace {

namespace vm = visualization_msgs::msg;

void set_color(gz::msgs::Color *c, const std_msgs::msg::ColorRGBA &src) {
  c->set_r(src.r);
  c->set_g(src.g);
  c->set_b(src.b);
  c->set_a(src.a);
}

// visualization_msgs/Marker.type -> gz::msgs::Marker::Type. Returns false for
// types we do not map (caller skips the marker).
bool map_type(int32_t in, gz::msgs::Marker::Type &out) {
  switch (in) {
    case vm::Marker::CUBE: out = gz::msgs::Marker::BOX; return true;
    case vm::Marker::SPHERE: out = gz::msgs::Marker::SPHERE; return true;
    case vm::Marker::CYLINDER: out = gz::msgs::Marker::CYLINDER; return true;
    case vm::Marker::LINE_STRIP: out = gz::msgs::Marker::LINE_STRIP; return true;
    case vm::Marker::LINE_LIST: out = gz::msgs::Marker::LINE_LIST; return true;
    case vm::Marker::POINTS: out = gz::msgs::Marker::POINTS; return true;
    case vm::Marker::TRIANGLE_LIST: out = gz::msgs::Marker::TRIANGLE_LIST; return true;
    case vm::Marker::TEXT_VIEW_FACING: out = gz::msgs::Marker::TEXT; return true;
    default: return false;
  }
}

gz::msgs::Marker::Action map_action(int32_t in) {
  switch (in) {
    case vm::Marker::DELETE: return gz::msgs::Marker::DELETE_MARKER;
    case vm::Marker::DELETEALL: return gz::msgs::Marker::DELETE_ALL;
    default: return gz::msgs::Marker::ADD_MODIFY;  // ADD == MODIFY
  }
}

// Fill a gz Marker from a ROS Marker. Returns false if the type is unmapped.
bool convert(const vm::Marker &in, gz::msgs::Marker &out) {
  out.set_ns(in.ns);
  out.set_id(in.id);
  out.set_action(map_action(in.action));

  if (out.action() == gz::msgs::Marker::ADD_MODIFY) {
    gz::msgs::Marker::Type t;
    if (!map_type(in.type, t)) return false;
    out.set_type(t);
  }

  auto *pos = out.mutable_pose()->mutable_position();
  pos->set_x(in.pose.position.x);
  pos->set_y(in.pose.position.y);
  pos->set_z(in.pose.position.z);
  auto *q = out.mutable_pose()->mutable_orientation();
  q->set_x(in.pose.orientation.x);
  q->set_y(in.pose.orientation.y);
  q->set_z(in.pose.orientation.z);
  q->set_w(in.pose.orientation.w == 0.0 ? 1.0 : in.pose.orientation.w);

  // only forward a fully-positive scale: a zero axis collapses the marker in gz,
  // and line markers leave scale unset (0,0,0) on purpose -> gz default (1,1,1).
  if (in.scale.x > 0.0 && in.scale.y > 0.0 && in.scale.z > 0.0) {
    out.mutable_scale()->set_x(in.scale.x);
    out.mutable_scale()->set_y(in.scale.y);
    out.mutable_scale()->set_z(in.scale.z);
  }

  // unlit fill: set ambient, diffuse and emissive so the colour shows regardless
  // of scene lighting.
  auto *mat = out.mutable_material();
  set_color(mat->mutable_ambient(), in.color);
  set_color(mat->mutable_diffuse(), in.color);
  set_color(mat->mutable_emissive(), in.color);

  for (const auto &p : in.points) {
    auto *gp = out.add_point();
    gp->set_x(p.x);
    gp->set_y(p.y);
    gp->set_z(p.z);
  }

  if (!in.text.empty()) out.set_text(in.text);

  // lifetime 0 = persists; otherwise expire (good for auto-clearing live frames).
  if (in.lifetime.sec != 0 || in.lifetime.nanosec != 0) {
    out.mutable_lifetime()->set_sec(in.lifetime.sec);
    out.mutable_lifetime()->set_nsec(in.lifetime.nanosec);
  }
  return true;
}

class MarkerBridge : public rclcpp::Node {
 public:
  MarkerBridge() : rclcpp::Node("ros_gz_marker_bridge") {
    topic_ = declare_parameter<std::string>("topic", "gz_markers");
    service_ = declare_parameter<std::string>("service", "/marker_array");
    reply_cb_ = [this](const gz::msgs::Boolean &rep, const bool result) {
      if (!result || !rep.data()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "gz %s request failed (result=%d data=%d)",
                             service_.c_str(), result, rep.data());
      }
    };
    sub_ = create_subscription<vm::MarkerArray>(
        topic_, rclcpp::QoS(10),
        std::bind(&MarkerBridge::on_markers, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "bridging ROS '%s' -> gz service '%s'",
                topic_.c_str(), service_.c_str());
  }

 private:
  void on_markers(const vm::MarkerArray::SharedPtr msg) {
    gz::msgs::Marker_V req;
    int skipped = 0;
    for (const auto &m : msg->markers) {
      if (!convert(m, *req.add_marker())) {
        req.mutable_marker()->RemoveLast();
        ++skipped;
      }
    }
    if (skipped) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "skipped %d marker(s) with unmapped type", skipped);
    }
    if (req.marker_size() == 0) return;
    node_.Request(service_, req, reply_cb_);
  }

  std::string topic_;
  std::string service_;
  std::function<void(const gz::msgs::Boolean &, const bool)> reply_cb_;
  gz::transport::Node node_;
  rclcpp::Subscription<vm::MarkerArray>::SharedPtr sub_;
};

}  // namespace

int main(int argc, char **argv) {
  // pin gz-transport to loopback (NixOS firewall drops the auto-picked iface);
  // don't clobber an explicit GZ_IP from the launch.
  setenv("GZ_IP", "127.0.0.1", 0);
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MarkerBridge>());
  rclcpp::shutdown();
  return 0;
}
