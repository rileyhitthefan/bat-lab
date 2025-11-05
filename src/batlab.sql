-- MySQL dump 10.13  Distrib 8.0.40, for Win64 (x86_64)
--
-- Host: localhost    Database: batlab_schema
-- ------------------------------------------------------
-- Server version	9.1.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `bats`
--

DROP TABLE IF EXISTS `bats`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `bats` (
  `idBats` int NOT NULL,
  `Abreviation` varchar(45) NOT NULL,
  `latin name` varchar(45) DEFAULT NULL,
  `common name` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`idBats`),
  UNIQUE KEY `idBats_UNIQUE` (`idBats`),
  UNIQUE KEY `Abreviation_UNIQUE` (`Abreviation`),
  UNIQUE KEY `latin name_UNIQUE` (`latin name`),
  UNIQUE KEY `common name_UNIQUE` (`common name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `bats`
--

LOCK TABLES `bats` WRITE;
/*!40000 ALTER TABLE `bats` DISABLE KEYS */;
/*!40000 ALTER TABLE `bats` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `call_library`
--

DROP TABLE IF EXISTS `call_library`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `call_library` (
  `idTraining` int NOT NULL,
  `files` blob,
  `bat` int DEFAULT NULL,
  `location` int DEFAULT NULL,
  PRIMARY KEY (`idTraining`),
  UNIQUE KEY `idTraining_UNIQUE` (`idTraining`),
  KEY `Bat_idx` (`bat`,`location`),
  CONSTRAINT `Bat_location_fk` FOREIGN KEY (`bat`, `location`) REFERENCES `location_bats` (`bat_id`, `location_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `call_library`
--

LOCK TABLES `call_library` WRITE;
/*!40000 ALTER TABLE `call_library` DISABLE KEYS */;
/*!40000 ALTER TABLE `call_library` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `location_bats`
--

DROP TABLE IF EXISTS `location_bats`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `location_bats` (
  `id` int NOT NULL AUTO_INCREMENT,
  `bat_id` int NOT NULL,
  `location_id` int NOT NULL,
  `lowest_frequency` int DEFAULT NULL,
  `highest_frequency` int DEFAULT NULL,
  `shape` blob,
  PRIMARY KEY (`id`),
  UNIQUE KEY `bat_location_unique` (`bat_id`,`location_id`),
  KEY `location_idx` (`location_id`),
  KEY `bat_idx` (`bat_id`),
  CONSTRAINT `fk_bat` FOREIGN KEY (`bat_id`) REFERENCES `bats` (`idBats`),
  CONSTRAINT `fk_location` FOREIGN KEY (`location_id`) REFERENCES `locations` (`idlocation`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `location_bats`
--

LOCK TABLES `location_bats` WRITE;
/*!40000 ALTER TABLE `location_bats` DISABLE KEYS */;
/*!40000 ALTER TABLE `location_bats` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `locations`
--

DROP TABLE IF EXISTS `locations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `locations` (
  `idlocation` int NOT NULL,
  `Area name` varchar(45) NOT NULL,
  `Latitude` decimal(6,0) NOT NULL,
  `Longitude` decimal(6,0) NOT NULL,
  `City` varchar(45) DEFAULT NULL,
  `Country` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`Area name`),
  UNIQUE KEY `idtable1_UNIQUE` (`idlocation`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `locations`
--

LOCK TABLES `locations` WRITE;
/*!40000 ALTER TABLE `locations` DISABLE KEYS */;
/*!40000 ALTER TABLE `locations` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping events for database 'batlab_schema'
--

--
-- Dumping routines for database 'batlab_schema'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-10-27 18:34:48
