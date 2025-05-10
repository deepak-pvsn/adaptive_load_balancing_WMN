#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/mesh-module.h"
#include "ns3/wifi-module.h"
#include "ns3/wifi-phy.h"
#include "ns3/applications-module.h"
#include <fstream>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("WirelessMeshCustomFinal");

// Global counters
static uint64_t g_bytesReceived   = 0;
static uint32_t g_packetsReceived = 0;
static uint32_t g_packetsSent     = 0;
static uint32_t g_drops           = 0;

// Previous‐interval snapshots
static uint64_t g_prevBytes       = 0;
static uint32_t g_prevReceived    = 0;
static uint32_t g_prevSent        = 0;
static uint32_t g_prevDrops       = 0;

//—— Trace callbacks ——
void RxCallback(Ptr<const Packet> pkt, const Address &)
{
  g_bytesReceived += pkt->GetSize();
  ++g_packetsReceived;
}
void TxCallback(Ptr<const Packet>)
{
  ++g_packetsSent;
}
void DropCallback(Ptr<const Packet>, WifiPhyRxfailureReason)
{
  ++g_drops;
}

//—— Periodic logging ——
void LogMetrics(Time interval)
{
  uint64_t bytesDelta = g_bytesReceived - g_prevBytes;
  uint32_t recvDelta  = g_packetsReceived - g_prevReceived;
  uint32_t sentDelta  = g_packetsSent     - g_prevSent;
  uint32_t dropsDelta = g_drops           - g_prevDrops;

  double bwMbps = double(bytesDelta * 8) / (1e6 * interval.GetSeconds());
  uint32_t pktLoss = dropsDelta;
  double pdr = sentDelta ? double(recvDelta) / sentDelta : 0.0;
  double cong = sentDelta ? double(dropsDelta) / sentDelta : 0.0;

  std::ofstream csv("simulation_data.csv", std::ios::app);
  csv << Simulator::Now().GetSeconds() << ","
      << bwMbps  << ","
      << pktLoss << ","
      << pdr     << ","
      << cong    << "\n";
  csv.close();

  g_prevBytes    = g_bytesReceived;
  g_prevReceived = g_packetsReceived;
  g_prevSent     = g_packetsSent;
  g_prevDrops    = g_drops;

  Simulator::Schedule(interval, &LogMetrics, interval);
}

int main(int argc, char *argv[])
{
  LogComponentEnable("WirelessMeshCustomFinal", LOG_LEVEL_INFO);

  // 1) CSV header
  std::ofstream header("simulation_data.csv");
  header << "time,bandwidth_utilization_Mbps,packet_loss_count,"
            "packet_delivery_ratio,congestion_ratio\n";
  header.close();

  // 2) Create 20 nodes
  NodeContainer nodes; 
  nodes.Create(20);

  // 3) Static grid mobility with tighter spacing (50 m)
  MobilityHelper mob;
  mob.SetPositionAllocator("ns3::GridPositionAllocator",
      "MinX", DoubleValue(0.0), "MinY", DoubleValue(0.0),
      "DeltaX", DoubleValue(50.0), "DeltaY", DoubleValue(50.0),
      "GridWidth", UintegerValue(5), "LayoutType", StringValue("RowFirst"));
  mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mob.Install(nodes);

  // 4) PHY & channel (boosted Tx power)
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
  YansWifiPhyHelper     phy;
  phy.SetChannel(channel.Create());
  phy.Set("TxPowerStart", DoubleValue(20.0));
  phy.Set("TxPowerEnd",   DoubleValue(20.0));

  // 5) Mesh stack with explicit standard & station manager :contentReference[oaicite:0]{index=0}
  MeshHelper mesh = MeshHelper::Default();
  mesh.SetStandard(WIFI_STANDARD_80211a);
  mesh.SetRemoteStationManager("ns3::ConstantRateWifiManager",
      "DataMode",    StringValue("OfdmRate6Mbps"),
      "ControlMode", StringValue("OfdmRate6Mbps"));
  mesh.SetStackInstaller("ns3::Dot11sStack");
  mesh.SetSpreadInterfaceChannels(MeshHelper::SPREAD_CHANNELS);

  NetDeviceContainer devices = mesh.Install(phy, nodes);

  // 6) Internet & IPv4
  InternetStackHelper internet;
  internet.Install(nodes);
  Ipv4AddressHelper ipv4;
  ipv4.SetBase("10.1.1.0","255.255.255.0");
  Ipv4InterfaceContainer ifs = ipv4.Assign(devices);

  // 7) Flow 1: node 0 → node 1
  uint16_t port1 = 9000;
  Address sinkAddr1(InetSocketAddress(ifs.GetAddress(1), port1));
  OnOffHelper onoff1("ns3::UdpSocketFactory", sinkAddr1);
  onoff1.SetAttribute("DataRate",   StringValue("1Mbps"));
  onoff1.SetAttribute("PacketSize", UintegerValue(1024));
  onoff1.SetAttribute("OnTime",     StringValue("ns3::UniformRandomVariable[Min=0.5|Max=1.5]"));
  onoff1.SetAttribute("OffTime",    StringValue("ns3::UniformRandomVariable[Min=0.5|Max=2.0]"));
  ApplicationContainer app1 = onoff1.Install(nodes.Get(0));
  app1.Start(Seconds(2.0)); app1.Stop(Seconds(602.0));

  // Mid-run rate bump at t=302 s
  Simulator::Schedule(Seconds(302.0), [&]() {
    onoff1.SetAttribute("DataRate", StringValue("3Mbps"));
    ApplicationContainer app1b = onoff1.Install(nodes.Get(0));
    app1b.Start(Seconds(302.0)); app1b.Stop(Seconds(602.0));
  });

  // Sink 1
  PacketSinkHelper sink1("ns3::UdpSocketFactory",
                         InetSocketAddress(Ipv4Address::GetAny(), port1));
  ApplicationContainer s1 = sink1.Install(nodes.Get(1));
  s1.Start(Seconds(1.0)); s1.Stop(Seconds(602.0));
  DynamicCast<PacketSink>(s1.Get(0))
      ->TraceConnectWithoutContext("Rx", MakeCallback(&RxCallback));  // :contentReference[oaicite:1]{index=1}

  // 8) Flow 2: node 2 → node 3
  uint16_t port2 = 9001;
  Address sinkAddr2(InetSocketAddress(ifs.GetAddress(3), port2));
  OnOffHelper onoff2("ns3::UdpSocketFactory", sinkAddr2);
  onoff2.SetAttribute("DataRate",   StringValue("500Kbps"));
  onoff2.SetAttribute("PacketSize", UintegerValue(512));
  onoff2.SetAttribute("OnTime",     StringValue("ns3::ExponentialRandomVariable[Mean=1.0]"));
  onoff2.SetAttribute("OffTime",    StringValue("ns3::ExponentialRandomVariable[Mean=2.0]"));
  ApplicationContainer app2 = onoff2.Install(nodes.Get(2));
  app2.Start(Seconds(5.0)); app2.Stop(Seconds(605.0));

  // Sink 2
  PacketSinkHelper sink2("ns3::UdpSocketFactory",
                         InetSocketAddress(Ipv4Address::GetAny(), port2));
  ApplicationContainer s2 = sink2.Install(nodes.Get(3));
  s2.Start(Seconds(4.0)); s2.Stop(Seconds(605.0));
  DynamicCast<PacketSink>(s2.Get(0))
      ->TraceConnectWithoutContext("Rx", MakeCallback(&RxCallback));  

  // 9) PHY‐level traces via Config :contentReference[oaicite:2]{index=2}
  Config::ConnectWithoutContext(
    "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/"
    "$ns3::YansWifiPhy/PhyTxEnd",
    MakeCallback(&TxCallback));
  Config::ConnectWithoutContext(
    "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/"
    "$ns3::YansWifiPhy/PhyRxDrop",
    MakeCallback(&DropCallback));

  // 10) Start logging & run
  Simulator::Schedule(Seconds(1.0), &LogMetrics, Seconds(1.0));        // :contentReference[oaicite:3]{index=3}
  Simulator::Stop(Seconds(605.0));
  Simulator::Run();
  Simulator::Destroy();
  return 0;
}
