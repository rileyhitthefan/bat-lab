SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- Schema
CREATE SCHEMA IF NOT EXISTS `batlab_schema` DEFAULT CHARACTER SET utf8;
USE `batlab_schema`;

-- Locations table
CREATE TABLE IF NOT EXISTS `Locations` (
  `area_name` VARCHAR(45) NOT NULL,
  `latitude` DECIMAL(6,3) NOT NULL,
  `longitude` DECIMAL(6,3) NOT NULL,
  PRIMARY KEY (`idlocation`),
  UNIQUE INDEX `area_name_UNIQUE` (`area_name` ASC)
) ENGINE=InnoDB;

-- Bats table
CREATE TABLE IF NOT EXISTS `Bats` (
  `abreviation` VARCHAR(45) NOT NULL,
  `latin_name` VARCHAR(45) NULL,
  `common_name` VARCHAR(45) NULL,
  PRIMARY KEY (`abreviation`),
  UNIQUE INDEX `latin_name_UNIQUE` (`latin_name` ASC),
  UNIQUE INDEX `abreviation_UNIQUE` (`abreviation` ASC),
  UNIQUE INDEX `common_name_UNIQUE` (`common_name` ASC)
) ENGINE=InnoDB;

-- Location_Bats table
CREATE TABLE IF NOT EXISTS `Location_Bats` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `bat_id` INT NOT NULL,
  `location_id` INT NOT NULL,
  `lowest_frequency` INT NULL,
  `highest_frequency` INT NULL,
  `shape` BLOB NULL,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `bat_location_unique` (`bat_id`, `location_id`),
  INDEX `location_idx` (`location_id`),
  INDEX `bat_idx` (`bat_id`),
  CONSTRAINT `fk_bat` FOREIGN KEY (`bat_id`)
    REFERENCES `Bats` (`idBats`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `fk_location` FOREIGN KEY (`location_id`)
    REFERENCES `Locations` (`idlocation`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB;

-- Call_Library table
CREATE TABLE IF NOT EXISTS `Call_Library` (
  `idTraining` INT NOT NULL,
  `files` BLOB NULL,
  `bat` INT NULL,
  `location` INT NULL,
  PRIMARY KEY (`idTraining`),
  UNIQUE INDEX `idTraining_UNIQUE` (`idTraining` ASC),
  INDEX `Bat_idx` (`bat`, `location`),
  CONSTRAINT `Bat_location_fk` FOREIGN KEY (`bat`, `location`)
    REFERENCES `Location_Bats` (`bat_id`, `location_id`)
    ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB;

SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
